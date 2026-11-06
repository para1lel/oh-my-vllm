use crate::planning::Choices;
use crate::planning::aligned::select_aligned;
use oh_my_vllm_kv_cache::group::RequestId;
use rustc_hash::FxHashMap;

pub(super) fn select_prefixes(
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
