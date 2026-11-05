use std::collections::{BTreeMap, BTreeSet};

use oh_my_vllm_kv_cache::hash::hash_block_tokens;
use oh_my_vllm_scheduler::{Request, output::ScheduledRequest};

use crate::error::{Error, Result};
use crate::protocol::{RankCompletion, RankSchedule};

#[derive(Default)]
pub(crate) struct RequestOwners {
    owners: BTreeMap<u64, u32>,
    pending: Option<(u64, Vec<RankSchedule>)>,
    available: [(usize, usize); 4],
}

impl RequestOwners {
    pub(crate) fn assign(&mut self, request_id: u64) -> u32 {
        if let Some(&rank) = self.owners.get(&request_id) {
            return rank;
        }
        let rank = (0..4)
            .max_by_key(|&rank| {
                let count = self
                    .owners
                    .values()
                    .filter(|&&owner| owner == rank as u32)
                    .count();
                (
                    std::cmp::Reverse(count),
                    self.available[rank].0.min(self.available[rank].1),
                    std::cmp::Reverse(rank),
                )
            })
            .unwrap() as u32;
        self.owners.insert(request_id, rank);
        rank
    }

    pub(crate) fn abort(&mut self, id: u64, ranks: &[u32]) -> Result<()> {
        if ranks != [0, 1, 2, 3] || self.pending.is_some() {
            return Err(Error::UnexpectedMessageType(
                "DEP4 abort has not completed on all ranks".into(),
            ));
        }
        self.owners.remove(&id);
        Ok(())
    }

    pub(crate) fn set_available(&mut self, rank: usize, fa: usize, state: usize) {
        self.available[rank] = (fa, state);
    }

    pub(crate) fn rank(&self, id: u64) -> usize {
        self.owners[&id] as usize
    }

    pub(crate) fn bind_cache(&mut self, request: &mut Request) {
        let rank = self.assign(request.id);
        request.cache_owner_rank = Some(rank);
        request.cache_salt = Some(hash_block_tokens(
            request.cache_salt,
            &[],
            &[0x64657034, u64::from(rank)],
        ));
    }

    fn owner(&self, id: u64) -> Result<u32> {
        self.owners.get(&id).copied().ok_or_else(|| {
            Error::UnexpectedMessageType(format!("DEP4 request {id} has no Rust owner"))
        })
    }

    pub(crate) fn schedule(
        &mut self,
        step: u64,
        scheduled: &[ScheduledRequest],
        finished: &[u64],
        preempted: &[u64],
    ) -> Result<Vec<RankSchedule>> {
        if self.pending.is_some() {
            return Err(Error::UnexpectedMessageType(
                "DEP4 previous step has not completed on all ranks".into(),
            ));
        }
        let mut plan: Vec<_> = (0..4)
            .map(|rank| RankSchedule {
                rank,
                request_ids: Vec::new(),
                finished_request_ids: Vec::new(),
                preempted_request_ids: Vec::new(),
            })
            .collect();
        let mut seen = BTreeSet::new();
        for row in scheduled {
            if !seen.insert(row.request_id) {
                return Err(Error::UnexpectedMessageType("duplicate DEP4 row".into()));
            }
            plan[self.owner(row.request_id)? as usize]
                .request_ids
                .push(row.request_id);
        }
        for (ids, is_finished) in [(finished, true), (preempted, false)] {
            let mut unique = BTreeSet::new();
            for &id in ids {
                if !unique.insert(id) || (is_finished && seen.contains(&id)) {
                    return Err(Error::UnexpectedMessageType(
                        "invalid DEP4 request lifecycle sets".into(),
                    ));
                }
                let local = &mut plan[self.owner(id)? as usize];
                if is_finished {
                    if preempted.contains(&id) {
                        return Err(Error::UnexpectedMessageType(
                            "finished DEP4 request is also preempted".into(),
                        ));
                    }
                    local.finished_request_ids.push(id);
                } else {
                    local.preempted_request_ids.push(id);
                }
            }
        }
        self.pending = Some((step, plan.clone()));
        Ok(plan)
    }

    pub(crate) fn complete(&mut self, step: u64, completions: &[RankCompletion]) -> Result<()> {
        let Some((pending_step, plan)) = &self.pending else {
            return Err(Error::UnexpectedMessageType(
                "no DEP4 step is pending".into(),
            ));
        };
        let mut ranks = BTreeSet::new();
        for completion in completions {
            if completion.step_id != step
                || *pending_step != step
                || completion.rank >= 4
                || !ranks.insert(completion.rank)
                || completion.schedule != plan[completion.rank as usize]
            {
                return Err(Error::UnexpectedMessageType(
                    "DEP4 completion differs from the Rust rank schedule".into(),
                ));
            }
        }
        if ranks.len() != 4 {
            return Err(Error::UnexpectedMessageType(
                "DEP4 completion is missing a rank".into(),
            ));
        }
        for local in plan {
            for id in &local.finished_request_ids {
                self.owners.remove(id);
            }
        }
        self.pending = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn completions(step_id: u64, plan: &[RankSchedule]) -> Vec<RankCompletion> {
        plan.iter()
            .map(|schedule| RankCompletion {
                rank: schedule.rank,
                step_id,
                schedule: schedule.clone(),
            })
            .collect()
    }

    #[test]
    fn prepared_owners_stay_balanced_when_private_state_pressure_differs() {
        let mut owners = RequestOwners::default();
        for id in 1..=63 {
            owners.assign(id);
        }
        for (rank, free_state) in [31, 15, 15, 22].into_iter().enumerate() {
            owners.set_available(rank, 50_000, free_state);
        }
        assert_eq!(owners.assign(64), 3);
        for id in 65..=128 {
            owners.assign(id);
        }
        for rank in 0..4 {
            assert_eq!(
                owners
                    .owners
                    .values()
                    .filter(|&&value| value == rank)
                    .count(),
                32
            );
        }
        assert_eq!(owners.assign(64), 3);
        owners.abort(1, &[0, 1, 2, 3]).unwrap();
        assert_eq!(owners.assign(129), 0);
    }

    #[test]
    fn zero_row_ranks_must_complete_before_owner_release() {
        let mut owners = RequestOwners::default();
        owners.assign(1);
        assert_eq!(owners.assign(2), 1);
        let plan = owners.schedule(1, &[], &[2], &[]).unwrap();
        assert_eq!(plan.len(), 4);
        assert!(plan.iter().all(|rank| rank.request_ids.is_empty()));
        let replies = completions(1, &plan);
        assert!(owners.complete(1, &replies[..3]).is_err());
        assert_eq!(owners.owner(2).unwrap(), 1);
        assert!(owners.schedule(2, &[], &[], &[]).is_err());
        let mut duplicate = replies.clone();
        duplicate[3] = duplicate[0].clone();
        assert!(owners.complete(1, &duplicate).is_err());
        let mut stale = replies.clone();
        stale[0].step_id = 0;
        assert!(owners.complete(1, &stale).is_err());
        owners.complete(1, &replies).unwrap();
        assert!(owners.owner(2).is_err());
    }

    #[test]
    fn preparation_abort_releases_owner_only_after_all_rank_acknowledgements() {
        let mut owners = RequestOwners::default();
        owners.assign(1);
        assert!(owners.abort(1, &[0, 1, 2]).is_err());
        assert!(owners.owner(1).is_ok());
        assert!(owners.abort(1, &[0, 1, 2, 2]).is_err());
        owners.abort(1, &[0, 1, 2, 3]).unwrap();
        assert!(owners.owner(1).is_err());
    }

    #[test]
    fn preemption_restart_keeps_owner_and_rank_local_prefix_identity() {
        let mut owners = RequestOwners::default();
        for id in 1..=5 {
            owners.assign(id);
        }
        let plan = owners.schedule(1, &[], &[], &[2]).unwrap();
        owners.complete(1, &completions(1, &plan)).unwrap();
        assert_eq!(owners.owner(2).unwrap(), 1);
        let mut requests = [1, 2, 5].map(|id| Request::new(id, vec![7; 256], 4, vec![]));
        for request in &mut requests {
            owners.bind_cache(request);
        }
        assert_ne!(requests[0].cache_salt, requests[1].cache_salt);
        assert_eq!(requests[0].cache_salt, requests[2].cache_salt);
        let plan = owners.schedule(2, &[], &[2], &[]).unwrap();
        let mut wrong = completions(2, &plan);
        wrong[1].schedule.finished_request_ids.clear();
        assert!(owners.complete(2, &wrong).is_err());
        assert_eq!(owners.owner(2).unwrap(), 1);
        owners.complete(2, &completions(2, &plan)).unwrap();
    }
}
