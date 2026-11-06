use crate::planning::Choices;
use oh_my_vllm_kv_cache::group::RequestId;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

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

#[derive(Clone, Copy)]
struct TokenRange {
    first: usize,
    last: usize,
    whole_blocks: bool,
    suffix: Option<i64>,
}

pub(super) fn select_dynamic(
    choices: &[Choices],
    budget: usize,
    block: usize,
    slots: usize,
    limited: bool,
    contributors: usize,
) -> FxHashMap<RequestId, usize> {
    let plan = DynamicPlan::new(budget, block, slots, limited, contributors);
    let mut previous = vec![-1i64; plan.width * (plan.slot_limit + 1)];
    // (decode count, whole blocks, prefill tokens, -suffix count).
    previous[0] = plan.unit - 1;
    let mut trace = Vec::new();
    for choice in choices.iter().rev() {
        let (next, selected) = plan.advance(choice, &previous, limited);
        trace.push(selected);
        previous = next;
    }
    plan.trace_counts(choices, &previous, &trace, limited)
}

struct DynamicPlan {
    budget: usize,
    block: usize,
    width: usize,
    slot_limit: usize,
    unit: i64,
    block_unit: i64,
    decode_unit: i64,
}

impl DynamicPlan {
    fn new(budget: usize, block: usize, slots: usize, limited: bool, contributors: usize) -> Self {
        let slot_limit = if limited {
            slots
        } else {
            0
        };
        let width = budget + 1;
        let unit = contributors as i64 + 1;
        let block_unit = (budget + 1) as i64 * unit;
        let decode_unit = (budget / block + 1) as i64 * block_unit;
        Self {
            budget,
            block,
            width,
            slot_limit,
            unit,
            block_unit,
            decode_unit,
        }
    }

    fn advance(&self, choice: &Choices, previous: &[i64], limited: bool) -> (Vec<i64>, Vec<u32>) {
        let charge = usize::from(limited && choice.waiting);
        let mut next = previous.to_vec();
        let mut selected = vec![0u32; previous.len()];
        let no_blocks = !choice.decode
            && (choice.offset + choice.ranges.last().map_or(0, |range| range.1))
                .saturating_sub(self.block * usize::from(choice.offset > 0))
                < self.block;
        let modulus = if choice.decode || no_blocks {
            1
        } else {
            self.block
        };
        let residues = modulus.min(self.width);
        for used in charge..=self.slot_limit {
            let source_base = (used - charge) * self.width;
            let target_base = used * self.width;
            for (first, last, whole_blocks, suffix) in self.ranges(choice, no_blocks) {
                let range = TokenRange {
                    first,
                    last,
                    whole_blocks,
                    suffix,
                };
                self.advance_range(
                    choice,
                    previous,
                    (&mut next, &mut selected),
                    (source_base, target_base),
                    (modulus, residues),
                    range,
                );
            }
        }
        (next, selected)
    }

    fn ranges<'a>(
        &'a self,
        choice: &'a Choices,
        no_blocks: bool,
    ) -> impl Iterator<Item = (usize, usize, bool, Option<i64>)> + 'a {
        choice.ranges.iter().flat_map(move |&(first, last)| {
            let ranges = if no_blocks {
                let boundary = self.block - choice.offset;
                [
                    (first, last.min(boundary - 1), false, Some(1)),
                    (first.max(boundary), last.min(boundary), false, Some(0)),
                    (first.max(boundary + 1), last, false, Some(1)),
                ]
            } else {
                let boundary = if choice.offset == 0 || choice.decode {
                    first
                } else {
                    self.block - choice.offset
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
        })
    }

    fn advance_range(
        &self,
        choice: &Choices,
        previous: &[i64],
        outputs: (&mut [i64], &mut [u32]),
        bases: (usize, usize),
        shape: (usize, usize),
        range: TokenRange,
    ) {
        let (next, selected) = outputs;
        let (source_base, target_base) = bases;
        let (modulus, residues) = shape;
        let mut queues = vec![VecDeque::<(usize, i64)>::new(); residues];
        let mut maximum = Maximum::new(residues);
        for target in range.first..=self.budget {
            let source = target - range.first;
            let residue = source % modulus;
            let score = previous[source_base + source];
            if score >= 0 {
                let key = self.source_key(choice, source, score, modulus, range.whole_blocks);
                let queue = &mut queues[residue];
                while queue.back().is_some_and(|(_, old)| *old < key) {
                    queue.pop_back();
                }
                queue.push_back((source, key));
                let &(index, key) = queue.front().unwrap();
                maximum.set(residue, (key, std::cmp::Reverse(index)));
            }
            if target > range.last {
                let expired = target - range.last - 1;
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
            let best = self.best_score(choice, target, range, &maximum, residues);
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

    fn source_key(
        &self,
        choice: &Choices,
        source: usize,
        score: i64,
        modulus: usize,
        whole_blocks: bool,
    ) -> i64 {
        score
            - (source / modulus) as i64
                * if choice.decode || !whole_blocks {
                    0
                } else {
                    self.block_unit
                }
            - if choice.decode {
                0
            } else {
                source as i64 * self.unit
            }
    }

    fn best_score(
        &self,
        choice: &Choices,
        target: usize,
        range: TokenRange,
        maximum: &Maximum,
        residues: usize,
    ) -> Entry {
        if choice.decode {
            let (key, source) = maximum.get(0, residues);
            (key.saturating_add(self.decode_unit), source)
        } else if let Some(penalty) = range.suffix {
            let (key, source) = maximum.get(0, residues);
            (
                key.saturating_add(target as i64 * self.unit - penalty),
                source,
            )
        } else {
            // floor((target-source+offset)/self.block) has one carry at the target residue.
            let position = target + choice.offset;
            let residue = position % self.block;
            let fixed = if range.whole_blocks {
                (position as i64 / self.block as i64 - i64::from(choice.offset > 0))
                    * self.block_unit
            } else {
                0
            } + target as i64 * self.unit;
            [
                (maximum.get(0, residue), 1),
                (maximum.get(residue, residue + 1), 0),
                (
                    maximum.get(residue + 1, residues),
                    if range.whole_blocks {
                        self.block_unit + 1
                    } else {
                        1
                    },
                ),
            ]
            .into_iter()
            .map(|((key, source), penalty)| (key.saturating_add(fixed - penalty), source))
            .max()
            .unwrap()
        }
    }

    fn trace_counts(
        &self,
        choices: &[Choices],
        previous: &[i64],
        trace: &[Vec<u32>],
        limited: bool,
    ) -> FxHashMap<RequestId, usize> {
        let objective = |index: usize| previous[index];
        let best = (0..previous.len())
            .filter(|index| previous[*index] >= 0)
            .map(objective)
            .max()
            .unwrap();
        let mut counts = Vec::new();
        for mut index in
            (0..previous.len()).filter(|index| previous[*index] >= 0 && objective(*index) == best)
        {
            let mut current = vec![0; choices.len()];
            for (i, choice) in choices.iter().enumerate() {
                let count = trace[choices.len() - 1 - i][index] as usize;
                current[i] = count;
                index -= count;
                if limited && choice.waiting && count > 0 {
                    index -= self.width;
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
}
