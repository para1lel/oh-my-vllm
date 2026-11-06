mod aligned;
mod dynamic;
mod fast;
mod prefixes;
mod queued;

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
    if let Some(counts) = fast::simple(&choices, budget, limited) {
        return counts;
    }
    if let Some(counts) = fast::resident_decodes(&choices, budget, block, slots) {
        return counts;
    }
    if let Some(counts) = fast::unbounded_slots(&choices, budget, block, slots, limited) {
        return counts;
    }
    if let Some(counts) = fast::single_supplier(&choices, budget, block, slots) {
        return counts;
    }
    if choices.iter().all(|choice| {
        choice.waiting
            && !choice.decode
            && choice.offset == 0
            && choice.ranges.len() == 1
            && choice.ranges[0].0 == 1
    }) && let Some(counts) = queued::select_queued(&choices, budget, block, slots)
    {
        return counts;
    }
    if choices.len() <= 8
        && choices.iter().all(|choice| {
            choice.ranges.len() == 1
                && choice.ranges[0].0 == 1
                && (choice.offset == 0 || choice.ranges[0].1 <= block - choice.offset)
        })
        && let Some(counts) = prefixes::select_prefixes(&choices, budget, block, slots)
    {
        return counts;
    }
    dynamic::select_dynamic(&choices, budget, block, slots, limited, contributors)
}
