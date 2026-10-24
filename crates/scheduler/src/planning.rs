use std::collections::VecDeque;

use oh_my_vllm_kv_cache::group::RequestId;
use rustc_hash::FxHashMap;

use crate::{Request, Scheduler, policy::CachePolicy};

#[derive(Clone)]
struct Choices {
    id: RequestId,
    waiting: bool,
    decode: bool,
    offset: usize,
    ranges: Vec<(usize, usize)>,
}

impl<P: CachePolicy> Scheduler<P> {
    pub(super) fn batch_quotas(&self) -> FxHashMap<RequestId, usize> {
        self.plan_quotas(self.config.max_num_batched_tokens, true)
    }

    pub(super) fn retry_waiting_quotas(
        &mut self,
        budget: usize,
        unselected: &mut VecDeque<Request>,
    ) -> FxHashMap<RequestId, usize> {
        unselected.append(&mut self.waiting);
        self.waiting = std::mem::take(unselected);
        self.plan_quotas(budget, false)
    }

    fn plan_quotas(&self, budget: usize, include_running: bool) -> FxHashMap<RequestId, usize> {
        let slots = self
            .config
            .max_num_seqs
            .saturating_sub(self.resident_count());
        let block = self.kv.prefill_block_size().max(1);
        let mut requests = self
            .running
            .iter()
            .filter(|_| include_running)
            .map(|request| (request, request.num_computed_tokens, false))
            .chain(
                self.waiting
                    .iter()
                    .filter(|_| slots > 0)
                    .map(|request| (request, self.kv.find_prefix(request).length, true)),
            )
            .filter(|(request, _, _)| !self.paused.contains(&request.id))
            .collect::<Vec<_>>();
        if let Some(index) = requests
            .iter()
            .position(|(request, _, _)| Some(request.id) == self.prefill_cursor)
        {
            let length = requests.len();
            requests.rotate_left((index + 1) % length);
        }
        let mut equivalent = FxHashMap::default();
        let choices = requests
            .into_iter()
            .filter_map(|(request, start, waiting)| {
                let decode = start >= Self::prefill_end(request);
                let maximum = budget.min(request.num_tokens_with_spec().saturating_sub(start));
                let ranges = if decode {
                    if maximum == 0 {
                        vec![]
                    } else {
                        vec![(1, maximum)]
                    }
                } else {
                    self.kv.prefill_ranges(request, start, maximum, budget)
                };
                if ranges.is_empty() {
                    return None;
                }
                let offset = if decode {
                    0
                } else {
                    start % block
                };
                if waiting {
                    // Later identical waiters cannot improve on the first `slots` in cursor order.
                    let copies = equivalent
                        .entry((decode, offset, ranges.clone()))
                        .or_insert(0);
                    *copies += 1;
                    if *copies > slots {
                        return None;
                    }
                }
                Some(Choices {
                    id: request.id,
                    waiting,
                    decode,
                    offset,
                    ranges,
                })
            })
            .collect();
        select(choices, budget, block, slots)
    }

    pub(super) fn prefill_end(request: &Request) -> usize {
        request
            .prompt_len
            .max(request.token_ids.len().saturating_sub(1))
    }
}

// Keys use the smaller source index on ties: the current, earlier request then gets more tokens.
type Entry = (i64, std::cmp::Reverse<usize>);
const EMPTY: Entry = (i64::MIN, std::cmp::Reverse(usize::MAX));

struct Maximum {
    leaves: usize,
    tree: Vec<Entry>,
}

impl Maximum {
    fn new(length: usize) -> Self {
        let leaves = length.next_power_of_two();
        Self {
            leaves,
            tree: vec![EMPTY; 2 * leaves],
        }
    }

    fn set(&mut self, index: usize, value: Entry) {
        let mut node = self.leaves + index;
        self.tree[node] = value;
        while node > 1 {
            node /= 2;
            self.tree[node] = self.tree[2 * node].max(self.tree[2 * node + 1]);
        }
    }

    fn get(&self, first: usize, last: usize) -> Entry {
        let mut first = first.min(self.leaves) + self.leaves;
        let mut last = last.min(self.leaves) + self.leaves;
        let mut value = EMPTY;
        while first < last {
            if first % 2 == 1 {
                value = value.max(self.tree[first]);
                first += 1;
            }
            if last % 2 == 1 {
                last -= 1;
                value = value.max(self.tree[last]);
            }
            first /= 2;
            last /= 2;
        }
        value
    }
}

fn select(
    choices: Vec<Choices>,
    budget: usize,
    block: usize,
    slots: usize,
) -> FxHashMap<RequestId, usize> {
    if budget == 0 || choices.is_empty() {
        return FxHashMap::default();
    }
    let waiting = choices.iter().filter(|choice| choice.waiting).count();
    let contributors = choices.len() - waiting + waiting.min(slots);
    let limited = waiting > slots;
    let maximum = |choice: &Choices| choice.ranges.last().map_or(0, |range| range.1);
    if !limited && choices.iter().map(maximum).sum::<usize>() <= budget {
        return choices
            .iter()
            .map(|choice| (choice.id, maximum(choice)))
            .collect();
    }
    if choices
        .iter()
        .all(|choice| !choice.decode && choice.offset == 0 && choice.ranges == [(1, budget)])
    {
        return choices
            .first()
            .map(|choice| (choice.id, budget))
            .into_iter()
            .collect();
    }
    if choices
        .iter()
        .all(|choice| choice.decode && !choice.waiting)
    {
        let mut left = budget;
        let mut counts = choices
            .iter()
            .map(|choice| {
                let count = maximum(choice).min(usize::from(left > 0));
                left -= count;
                (choice.id, count)
            })
            .collect::<Vec<_>>();
        for (choice, (_, count)) in choices.iter().zip(&mut counts) {
            if *count > 0 {
                let extra = (maximum(choice) - *count).min(left);
                *count += extra;
                left -= extra;
            }
        }
        return counts.into_iter().collect();
    }
    let decodes = choices.iter().filter(|choice| choice.decode).count();
    if decodes > 0
        && choices
            .iter()
            .all(|choice| !choice.decode || !choice.waiting)
    {
        // Resident decodes need one row each. Extra verification rows use only
        // the budget left after the prefill objectives have been satisfied.
        let reserved = budget.min(decodes);
        let prefills = choices
            .iter()
            .filter(|choice| !choice.decode)
            .cloned()
            .collect();
        let mut counts = select(prefills, budget - reserved, block, slots);
        let mut extra = budget - reserved - counts.values().sum::<usize>();
        for choice in choices.iter().filter(|choice| choice.decode).take(reserved) {
            let count = 1 + (maximum(choice) - 1).min(extra);
            extra -= count - 1;
            counts.insert(choice.id, count);
        }
        return counts;
    }
    if limited {
        let mut waiting = choices
            .iter()
            .filter(|choice| choice.waiting)
            .collect::<Vec<_>>();
        // Without a binding token budget the objective is additive. Stable order
        // keeps cursor priority when two waiters make the same contribution.
        waiting.sort_by_key(|choice| {
            let count = maximum(choice);
            std::cmp::Reverse(if choice.decode {
                (1, 0, 0, std::cmp::Reverse(false))
            } else {
                (
                    0,
                    (choice.offset + count).saturating_sub(block * usize::from(choice.offset > 0))
                        / block,
                    count,
                    std::cmp::Reverse(!(choice.offset + count).is_multiple_of(block)),
                )
            })
        });
        let selected = choices
            .iter()
            .filter(|choice| !choice.waiting)
            .chain(waiting.into_iter().take(slots))
            .collect::<Vec<_>>();
        if selected
            .iter()
            .try_fold(budget, |left, choice| left.checked_sub(maximum(choice)))
            .is_some()
        {
            return selected
                .into_iter()
                .map(|choice| (choice.id, maximum(choice)))
                .collect();
        }
    }
    let whole = budget / block * block;
    if whole > 0
        && choices.iter().all(|choice| !choice.decode)
        && let Some(index) = choices.iter().position(|choice| {
            !choice.waiting
                && choice.offset == 0
                && choice.ranges.len() == 1
                && choice.ranges[0].0 == 1
                && maximum(choice) >= whole
        })
        && choices.iter().enumerate().all(|(i, choice)| {
            i == index
                || (choice.offset + maximum(choice))
                    .saturating_sub(block * usize::from(choice.offset > 0))
                    < block
        })
    {
        // Only this resident can supply whole blocks. Fix their optimal count;
        // the remaining budget still decides all suffix and cursor ties.
        let mut remaining = choices.clone();
        remaining[index].ranges[0].1 -= whole;
        for choice in &mut remaining {
            for (_, last) in &mut choice.ranges {
                *last = (*last).min(budget - whole);
            }
            choice.ranges.retain(|(first, last)| first <= last);
        }
        remaining.retain(|choice| !choice.ranges.is_empty());
        let mut counts = select(remaining, budget - whole, block, slots);
        *counts.entry(choices[index].id).or_default() += whole;
        return counts;
    }
    if choices.iter().all(|choice| {
        choice.waiting
            && !choice.decode
            && choice.offset == 0
            && choice.ranges.len() == 1
            && choice.ranges[0].0 == 1
    }) && let Some(counts) = select_queued(&choices, budget, block, slots)
    {
        return counts;
    }
    if choices.len() <= 8
        && choices.iter().all(|choice| {
            choice.ranges.len() == 1
                && choice.ranges[0].0 == 1
                && (choice.offset == 0 || choice.ranges[0].1 <= block - choice.offset)
        })
        && let Some(counts) = select_prefixes(&choices, budget, block, slots)
    {
        return counts;
    }
    let slot_limit = if limited {
        slots
    } else {
        0
    };
    let width = budget + 1;
    let states = width * (slot_limit + 1);
    let unit = contributors as i64 + 1;
    let block_unit = (budget + 1) as i64 * unit;
    let decode_unit = (budget / block + 1) as i64 * block_unit;
    // (decode count, whole blocks, prefill tokens, -suffix count).
    let mut previous = vec![-1i64; states];
    previous[0] = unit - 1;
    let mut trace = Vec::new();
    for choice in choices.iter().rev() {
        let charge = usize::from(limited && choice.waiting);
        let mut next = previous.clone();
        let mut selected = vec![0u32; states];
        let no_blocks = !choice.decode
            && (choice.offset + maximum(choice))
                .saturating_sub(block * usize::from(choice.offset > 0))
                < block;
        let modulus = if choice.decode || no_blocks {
            1
        } else {
            block
        };
        let residues = modulus.min(width);
        for used in charge..=slot_limit {
            let source_base = (used - charge) * width;
            let target_base = used * width;
            let ranges = choice.ranges.iter().flat_map(|&(first, last)| {
                let ranges = if no_blocks {
                    let boundary = block - choice.offset;
                    [
                        (first, last.min(boundary - 1), false, Some(1)),
                        (first.max(boundary), last.min(boundary), false, Some(0)),
                        (first.max(boundary + 1), last, false, Some(1)),
                    ]
                } else {
                    let boundary = if choice.offset == 0 || choice.decode {
                        first
                    } else {
                        block - choice.offset
                    };
                    [
                        (first, last.min(boundary - 1), choice.offset == 0, None),
                        (first.max(boundary), last, true, None),
                        (1, 0, false, None),
                    ]
                };
                ranges
                    .into_iter()
                    .filter(|(first, last, _, _)| first <= last)
            });
            for (first, last, whole_blocks, suffix) in ranges {
                let mut queues = vec![VecDeque::<(usize, i64)>::new(); residues];
                let mut maximum = Maximum::new(residues);
                for target in first..=budget {
                    let source = target - first;
                    let residue = source % modulus;
                    let score = previous[source_base + source];
                    if score >= 0 {
                        let key = score
                            - (source / modulus) as i64
                                * if choice.decode || !whole_blocks {
                                    0
                                } else {
                                    block_unit
                                }
                            - if choice.decode {
                                0
                            } else {
                                source as i64 * unit
                            };
                        let queue = &mut queues[residue];
                        while queue.back().is_some_and(|(_, old)| *old < key) {
                            queue.pop_back();
                        }
                        queue.push_back((source, key));
                        let &(index, key) = queue.front().unwrap();
                        maximum.set(residue, (key, std::cmp::Reverse(index)));
                    }
                    if target > last {
                        let expired = target - last - 1;
                        let residue = expired % modulus;
                        let queue = &mut queues[residue];
                        if queue.front().is_some_and(|(index, _)| *index == expired) {
                            queue.pop_front();
                            maximum.set(
                                residue,
                                queue
                                    .front()
                                    .map_or(EMPTY, |&(index, key)| (key, std::cmp::Reverse(index))),
                            );
                        }
                    }
                    let best = if choice.decode {
                        let (key, source) = maximum.get(0, residues);
                        (key.saturating_add(decode_unit), source)
                    } else if let Some(penalty) = suffix {
                        let (key, source) = maximum.get(0, residues);
                        (key.saturating_add(target as i64 * unit - penalty), source)
                    } else {
                        // floor((target-source+offset)/block) has one carry at the target residue.
                        let position = target + choice.offset;
                        let residue = position % block;
                        let fixed = if whole_blocks {
                            (position as i64 / block as i64 - i64::from(choice.offset > 0))
                                * block_unit
                        } else {
                            0
                        } + target as i64 * unit;
                        [
                            (maximum.get(0, residue), 1),
                            (maximum.get(residue, residue + 1), 0),
                            (
                                maximum.get(residue + 1, residues),
                                if whole_blocks {
                                    block_unit + 1
                                } else {
                                    1
                                },
                            ),
                        ]
                        .into_iter()
                        .map(|((key, source), penalty)| {
                            (key.saturating_add(fixed - penalty), source)
                        })
                        .max()
                        .unwrap()
                    };
                    let count = target - best.1.0.min(target);
                    let index = target_base + target;
                    if best.0 >= 0
                        && (best.0 > next[index]
                            || (best.0 == next[index] && count > selected[index] as usize))
                    {
                        next[index] = best.0;
                        selected[index] = count as u32;
                    }
                }
            }
        }
        trace.push(selected);
        previous = next;
    }
    let objective = |index: usize| previous[index];
    let best = (0..states)
        .filter(|index| previous[*index] >= 0)
        .map(objective)
        .max()
        .unwrap();
    let mut counts = Vec::new();
    for mut index in (0..states).filter(|index| previous[*index] >= 0 && objective(*index) == best)
    {
        let mut current = vec![0; choices.len()];
        for (i, choice) in choices.iter().enumerate() {
            let count = trace[choices.len() - 1 - i][index] as usize;
            current[i] = count;
            index -= count;
            if limited && choice.waiting && count > 0 {
                index -= width;
            }
        }
        if current > counts {
            counts = current;
        }
    }
    choices
        .iter()
        .zip(counts)
        .map(|(choice, count)| (choice.id, count))
        .collect()
}

fn top_sums(capacities: &[usize], slots: usize) -> Vec<Vec<usize>> {
    let mut top = Vec::new();
    let mut sums = vec![vec![0]; capacities.len() + 1];
    for i in (0..capacities.len()).rev() {
        let position = top.partition_point(|value| *value >= capacities[i]);
        top.insert(position, capacities[i]);
        top.truncate(slots);
        sums[i].extend(top.iter().scan(0, |sum, value| {
            *sum += value;
            Some(*sum)
        }));
    }
    sums
}

fn select_queued(
    choices: &[Choices],
    budget: usize,
    block: usize,
    slots: usize,
) -> Option<FxHashMap<RequestId, usize>> {
    let capacities = choices
        .iter()
        .map(|choice| choice.ranges[0].1)
        .collect::<Vec<_>>();
    let sum = |sums: &[Vec<usize>], i: usize, count: usize| sums[i][count.min(sums[i].len() - 1)];
    let mut result = FxHashMap::default();
    if capacities.iter().all(|count| *count < block) {
        let sums = top_sums(&capacities, slots);
        let mut tokens = budget.min(sum(&sums, 0, slots));
        let mut needed = (0..=slots).find(|count| sum(&sums, 0, *count) >= tokens)?;
        for (i, choice) in choices.iter().enumerate() {
            if needed == 0 {
                break;
            }
            let count = capacities[i].min(tokens - (needed - 1));
            if count > 0 && sum(&sums, i + 1, needed - 1) >= tokens - count {
                result.insert(choice.id, count);
                tokens -= count;
                needed -= 1;
            }
        }
        return Some(result);
    }
    let blocks = capacities
        .iter()
        .map(|count| count / block)
        .collect::<Vec<_>>();
    let sums = top_sums(&blocks, slots);
    let mut remaining_blocks = budget / block;
    let mut tail = budget % block;
    let mut available = slots;
    // With zero or one suffix, feasibility needs only the largest remaining block capacities.
    let feasible = |first: usize, needed: usize, slots: usize, tail: usize| {
        if tail == 0 {
            return sum(&sums, first, slots) >= needed;
        }
        if slots == 0 {
            return false;
        }
        let other = sum(&sums, first, slots - 1);
        (first..choices.len()).any(|i| {
            if capacities[i] < tail {
                return false;
            }
            let mut capacity = other;
            if slots > 1 && blocks[i] >= other - sum(&sums, first, slots - 2) {
                capacity = capacity - blocks[i] + sum(&sums, first, slots) - other;
            }
            capacity + (capacities[i] - tail) / block >= needed
        })
    };
    if !feasible(0, remaining_blocks, available, tail) {
        return None;
    }
    for (i, choice) in choices.iter().enumerate() {
        if available == 0 {
            break;
        }
        let mut count = 0;
        let whole = remaining_blocks.min(blocks[i]);
        if whole > 0 && feasible(i + 1, remaining_blocks - whole, available - 1, tail) {
            count = whole * block;
        }
        if tail > 0 && capacities[i] >= tail {
            let whole = remaining_blocks.min((capacities[i] - tail) / block);
            if feasible(i + 1, remaining_blocks - whole, available - 1, 0) {
                count = count.max(whole * block + tail);
            }
        }
        if count > 0 {
            remaining_blocks -= count / block;
            if count % block > 0 {
                tail = 0;
            }
            available -= 1;
            result.insert(choice.id, count);
        }
    }
    Some(result)
}

fn select_prefixes(
    choices: &[Choices],
    budget: usize,
    block: usize,
    slots: usize,
) -> Option<FxHashMap<RequestId, usize>> {
    let leading = choices
        .iter()
        .filter(|choice| choice.offset > 0 && choice.offset + choice.ranges[0].1 == block)
        .map(|choice| choice.id)
        .collect::<Vec<_>>();
    let mut best = None;
    for mask in 0..1usize << leading.len() {
        let mut tokens = 0;
        let mut admitted = 0;
        let mut counts = FxHashMap::default();
        let mut remaining = Vec::new();
        for choice in choices {
            let endpoint = leading.iter().position(|id| *id == choice.id);
            if endpoint.is_some_and(|bit| mask & (1 << bit) != 0) {
                let count = choice.ranges[0].1;
                counts.insert(choice.id, count);
                tokens += count;
                admitted += usize::from(choice.waiting);
            } else {
                let mut choice = choice.clone();
                choice.offset = 0;
                choice.ranges[0].1 -= usize::from(endpoint.is_some());
                if choice.ranges[0].1 > 0 {
                    remaining.push(choice);
                }
            }
        }
        if tokens > budget || admitted > slots {
            continue;
        }
        let slots = slots - admitted;
        let limited = remaining.iter().filter(|choice| choice.waiting).count() > slots;
        counts.extend(select_aligned(
            &remaining,
            budget - tokens,
            block,
            slots,
            limited,
        )?);
        let values = choices
            .iter()
            .map(|choice| counts.get(&choice.id).copied().unwrap_or(0))
            .collect::<Vec<_>>();
        let mut decodes = 0;
        let mut blocks = 0;
        let mut prefill = 0;
        let mut suffixes = 0;
        for (choice, &count) in choices.iter().zip(&values) {
            decodes += usize::from(choice.decode && count > 0);
            if !choice.decode && count > 0 {
                prefill += count;
                blocks += (count + choice.offset)
                    .saturating_sub(block * usize::from(choice.offset > 0))
                    / block;
                suffixes += usize::from(!(count + choice.offset).is_multiple_of(block));
            }
        }
        let score = (
            decodes,
            blocks,
            prefill,
            std::cmp::Reverse(suffixes),
            values,
        );
        if best.as_ref().is_none_or(|(old, _)| score > *old) {
            best = Some((score, counts));
        }
    }
    best.map(|(_, counts)| counts)
}

fn select_aligned(
    choices: &[Choices],
    budget: usize,
    block: usize,
    slots: usize,
    limited: bool,
) -> Option<FxHashMap<RequestId, usize>> {
    let decodes = choices.iter().filter(|choice| choice.decode).count();
    let suffixes = choices.len() - decodes;
    let blocks = budget / block;
    let admissions = if limited {
        slots
    } else {
        0
    };
    let shape = [decodes + 1, blocks + 1, suffixes + 1, admissions + 1];
    let states = shape.iter().product::<usize>();
    if states > 65536 {
        return None;
    }
    let index =
        |d: usize, b: usize, s: usize, w: usize| ((d * shape[1] + b) * shape[2] + s) * shape[3] + w;
    // For fixed decode/block/suffix counts, all token counts between the minimum
    // and the stored maximum are feasible. Keep this capacity, rather than token states.
    let options = choices
        .iter()
        .map(|choice| {
            let maximum = choice.ranges[0].1;
            let mut options = vec![(0, 0, 0, 0)];
            if choice.decode {
                options.push((1, 0, 0, maximum));
            } else {
                for b in 0..=maximum / block {
                    if b > 0 {
                        options.push((0, b, 0, b * block));
                    }
                    let end = maximum.min(b * block + block - 1);
                    if end > b * block {
                        options.push((0, b, 1, end));
                    }
                }
            }
            options
        })
        .collect::<Vec<_>>();
    let mut suffix = vec![vec![usize::MAX; states]];
    suffix[0][0] = 0;
    for (choice, options) in choices.iter().zip(&options).rev() {
        let previous = suffix.last().unwrap();
        let mut next = vec![usize::MAX; states];
        for d in 0..=decodes {
            for b in 0..=blocks {
                for s in 0..=suffixes {
                    for w in 0..=admissions {
                        let capacity = previous[index(d, b, s, w)];
                        if capacity == usize::MAX {
                            continue;
                        }
                        for &(add_d, add_b, add_s, count) in options {
                            let add_w = usize::from(limited && choice.waiting && count > 0);
                            let (d, b, s, w) = (d + add_d, b + add_b, s + add_s, w + add_w);
                            if d > decodes
                                || b > blocks
                                || s > suffixes
                                || w > admissions
                                || d + b * block + s > budget
                            {
                                continue;
                            }
                            let next = &mut next[index(d, b, s, w)];
                            let capacity = (capacity
                                + if add_d > 0 {
                                    0
                                } else {
                                    count
                                })
                            .min(budget - d);
                            if *next == usize::MAX || capacity > *next {
                                *next = capacity;
                            }
                        }
                    }
                }
            }
        }
        suffix.push(next);
    }
    let capacities = suffix.last().unwrap();
    let mut best = (0, 0, 0, std::cmp::Reverse(0));
    for d in 0..=decodes {
        for b in 0..=blocks {
            for s in 0..=suffixes {
                for w in 0..=admissions {
                    let capacity = capacities[index(d, b, s, w)];
                    if capacity != usize::MAX {
                        best = best.max((d, b, capacity, std::cmp::Reverse(s)));
                    }
                }
            }
        }
    }
    let (mut d, mut b, mut tokens, std::cmp::Reverse(mut s)) = best;
    let mut remaining_budget = budget;
    let mut available = admissions;
    let mut result = FxHashMap::default();
    for (i, (choice, options)) in choices.iter().zip(&options).enumerate() {
        let tail = &suffix[choices.len() - i - 1];
        let mut selected: Option<(usize, usize, usize, usize, usize)> = None;
        for &(take_d, take_b, take_s, maximum) in options {
            let charge = usize::from(limited && choice.waiting && maximum > 0);
            if take_d > d || take_b > b || take_s > s || charge > available {
                continue;
            }
            let minimum = take_d + take_b * block + take_s;
            let remainder_min = (b - take_b) * block + s - take_s;
            if remainder_min > tokens {
                continue;
            }
            let count = if take_d > 0 {
                maximum.min(remaining_budget - (d - take_d) - tokens)
            } else {
                maximum.min(tokens - remainder_min)
            };
            if count < minimum {
                continue;
            }
            for used in 0..=available - charge {
                let capacity = tail[index(d - take_d, b - take_b, s - take_s, used)];
                let prefill = if take_d > 0 {
                    0
                } else {
                    count
                };
                if capacity != usize::MAX
                    && tokens - prefill <= capacity
                    && selected.is_none_or(|(old, ..)| count > old)
                {
                    selected = Some((count, take_d, take_b, take_s, charge));
                }
            }
        }
        let (count, take_d, take_b, take_s, charge) =
            selected.expect("aligned capacity is feasible");
        result.insert(choice.id, count);
        d -= take_d;
        b -= take_b;
        s -= take_s;
        if take_d == 0 {
            tokens -= count;
        }
        remaining_budget -= count;
        available -= charge;
    }
    Some(result)
}
