use crate::planning::Choices;
use oh_my_vllm_kv_cache::group::RequestId;
use rustc_hash::FxHashMap;

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

pub(super) fn select_queued(
    choices: &[Choices],
    budget: usize,
    block: usize,
    slots: usize,
) -> Option<FxHashMap<RequestId, usize>> {
    let capacities = choices
        .iter()
        .map(|choice| choice.ranges[0].1)
        .collect::<Vec<_>>();
    let mut result = FxHashMap::default();
    if capacities.iter().all(|count| *count < block) {
        return select_tails(choices, &capacities, budget, slots);
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

fn sum(sums: &[Vec<usize>], i: usize, count: usize) -> usize {
    sums[i][count.min(sums[i].len() - 1)]
}

fn select_tails(
    choices: &[Choices],
    capacities: &[usize],
    budget: usize,
    slots: usize,
) -> Option<FxHashMap<RequestId, usize>> {
    let mut result = FxHashMap::default();

    let sums = top_sums(capacities, slots);
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
    Some(result)
}
