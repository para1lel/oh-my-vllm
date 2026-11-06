use crate::planning::{Choices, select};
use oh_my_vllm_kv_cache::group::RequestId;
use rustc_hash::FxHashMap;

fn maximum(choice: &Choices) -> usize {
    choice.ranges.last().map_or(0, |range| range.1)
}

pub(super) fn simple(
    choices: &[Choices],
    budget: usize,
    limited: bool,
) -> Option<FxHashMap<RequestId, usize>> {
    if !limited && choices.iter().map(maximum).sum::<usize>() <= budget {
        return Some(
            choices
                .iter()
                .map(|choice| (choice.id, maximum(choice)))
                .collect(),
        );
    }
    if choices
        .iter()
        .all(|choice| !choice.decode && choice.offset == 0 && choice.ranges == [(1, budget)])
    {
        return Some(
            choices
                .first()
                .map(|choice| (choice.id, budget))
                .into_iter()
                .collect(),
        );
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
        return Some(counts.into_iter().collect());
    }
    None
}

pub(super) fn resident_decodes(
    choices: &[Choices],
    budget: usize,
    block: usize,
    slots: usize,
) -> Option<FxHashMap<RequestId, usize>> {
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
        return Some(counts);
    }
    None
}

pub(super) fn unbounded_slots(
    choices: &[Choices],
    budget: usize,
    block: usize,
    slots: usize,
    limited: bool,
) -> Option<FxHashMap<RequestId, usize>> {
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
            return Some(
                selected
                    .into_iter()
                    .map(|choice| (choice.id, maximum(choice)))
                    .collect(),
            );
        }
    }
    None
}

pub(super) fn single_supplier(
    choices: &[Choices],
    budget: usize,
    block: usize,
    slots: usize,
) -> Option<FxHashMap<RequestId, usize>> {
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
        let mut remaining = choices.to_vec();
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
        return Some(counts);
    }
    None
}
