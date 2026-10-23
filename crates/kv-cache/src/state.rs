use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard};

use oh_my_vllm_model_contract::{CacheSchema, CacheSource, ContractError, ParallelPlan};
use serde::{Deserialize, Serialize};

use crate::BlockPool;

mod selection;

type PoolKey = (String, u32);

#[derive(Debug, thiserror::Error)]
pub enum StateError {
    #[error(transparent)]
    Contract(#[from] ContractError),
    #[error("invalid state reservation: {0}")]
    Invalid(&'static str),
    #[error("state arena is poisoned")]
    Poisoned,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocationRequest {
    pub group: String,
    pub rank: u32,
    pub blocks: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateBlock {
    pub group: String,
    pub rank: u32,
    pub slot: u32,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransactionKey {
    pub epoch: u64,
    pub step_id: u64,
    pub transaction_id: u64,
    pub request_id: u64,
}

struct Pool {
    blocks: BlockPool,
    generations: Vec<u64>,
    next_generation: u64,
}

struct Inner {
    epoch: u64,
    last_step: u64,
    next_transaction: u64,
    sources: BTreeMap<String, String>,
    pools: BTreeMap<PoolKey, Pool>,
}

pub struct StateArena {
    inner: Arc<Mutex<Inner>>,
}

pub struct StorageLease {
    inner: Arc<Mutex<Inner>>,
    epoch: u64,
    blocks: Vec<StateBlock>,
}

pub struct StateTransaction {
    key: TransactionKey,
    writes: StorageLease,
    reads: StorageLease,
}

fn lock(inner: &Mutex<Inner>) -> Result<MutexGuard<'_, Inner>, StateError> {
    inner.lock().map_err(|_| StateError::Poisoned)
}

fn sources(schema: &CacheSchema) -> Result<BTreeMap<String, String>, StateError> {
    let groups: BTreeMap<_, _> = schema
        .groups
        .iter()
        .map(|group| (&group.name, group))
        .collect();
    let mut result = BTreeMap::new();
    for group in &schema.groups {
        let mut owner = group;
        while let CacheSource::Shared { producer } = &owner.source {
            owner = groups[producer];
            if owner.kind != group.kind
                || owner.block_tokens != group.block_tokens
                || owner.storage != group.storage
                || owner.capacity_blocks != group.capacity_blocks
            {
                return Err(StateError::Invalid(
                    "shared storage geometry differs from its producer",
                ));
            }
        }
        result.insert(group.name.clone(), owner.name.clone());
    }
    Ok(result)
}

fn pool_capacity(capacity: u64) -> Result<u32, StateError> {
    let capacity = u32::try_from(capacity)
        .map_err(|_| StateError::Invalid("pool capacity does not fit u32"))?;
    if !(2..=u32::MAX - 2).contains(&capacity) {
        return Err(StateError::Invalid(
            "pool capacity must leave space for queue sentinels",
        ));
    }
    Ok(capacity)
}

impl StateArena {
    pub fn new(
        epoch: u64,
        schema: &CacheSchema,
        parallel: &ParallelPlan,
        max_slots: u64,
    ) -> Result<Self, StateError> {
        schema.validate()?;
        parallel.validate()?;
        if epoch == 0 {
            return Err(StateError::Invalid("epoch must be positive"));
        }
        let sources = sources(schema)?;
        let mut layouts = BTreeMap::new();
        let mut total = 0_u64;
        for group in &schema.groups {
            if !matches!(group.source, CacheSource::Owned {}) {
                continue;
            }
            let capacity = pool_capacity(group.capacity_blocks)?;
            for &rank in &parallel.ranks {
                total = total
                    .checked_add(u64::from(capacity))
                    .filter(|&value| value <= max_slots)
                    .ok_or(StateError::Invalid("logical slot budget is not sufficient"))?;
                layouts.insert((group.name.clone(), rank), capacity);
            }
        }
        let pools = layouts
            .into_iter()
            .map(|(key, capacity)| {
                (
                    key,
                    Pool {
                        blocks: BlockPool::new(capacity, false),
                        generations: vec![0; capacity as usize],
                        next_generation: 0,
                    },
                )
            })
            .collect();
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                epoch,
                last_step: 0,
                next_transaction: 0,
                sources,
                pools,
            })),
        })
    }

    pub fn free_blocks(&self, group: &str, rank: u32) -> Result<usize, StateError> {
        let inner = lock(&self.inner)?;
        let source = inner
            .sources
            .get(group)
            .ok_or(StateError::Invalid("unknown cache group"))?;
        let pool = inner
            .pools
            .get(&(source.clone(), rank))
            .ok_or(StateError::Invalid("unknown owner rank"))?;
        Ok(pool.blocks.num_free_blocks())
    }

    pub fn reserve(
        &self,
        step_id: u64,
        request_id: u64,
        allocations: &[AllocationRequest],
        reads: &[&StorageLease],
    ) -> Result<StateTransaction, StateError> {
        if reads
            .iter()
            .any(|lease| !Arc::ptr_eq(&self.inner, &lease.inner))
        {
            return Err(StateError::Invalid(
                "read lease has a different arena owner",
            ));
        }
        let read_blocks: Vec<_> = reads
            .iter()
            .flat_map(|lease| lease.blocks.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut inner = lock(&self.inner)?;
        if step_id == 0 || step_id < inner.last_step {
            return Err(StateError::Invalid("zero or stale step identity"));
        }
        let transaction_id = inner
            .next_transaction
            .checked_add(1)
            .ok_or(StateError::Invalid("transaction identity exhausted"))?;
        validate_reads(&inner, &read_blocks)?;
        let plan = allocation_plan(&inner, allocations)?;
        pin_blocks(&mut inner, &read_blocks);
        let mut writes = Vec::new();
        for (key, count) in plan {
            let pool = inner.pools.get_mut(&key).expect("validated pool");
            pool.next_generation += 1;
            let ids = pool
                .blocks
                .get_new_blocks(count)
                .expect("all capacities validated while locked");
            for slot in ids {
                pool.generations[slot as usize] = pool.next_generation;
                writes.push(StateBlock {
                    group: key.0.clone(),
                    rank: key.1,
                    slot,
                    generation: pool.next_generation,
                });
            }
        }
        inner.last_step = step_id;
        inner.next_transaction = transaction_id;
        let epoch = inner.epoch;
        Ok(StateTransaction {
            key: TransactionKey {
                epoch,
                step_id,
                transaction_id,
                request_id,
            },
            writes: StorageLease {
                inner: Arc::clone(&self.inner),
                epoch,
                blocks: writes,
            },
            reads: StorageLease {
                inner: Arc::clone(&self.inner),
                epoch,
                blocks: read_blocks,
            },
        })
    }
}

fn allocation_plan(
    inner: &Inner,
    allocations: &[AllocationRequest],
) -> Result<BTreeMap<PoolKey, usize>, StateError> {
    let mut plan = BTreeMap::new();
    for request in allocations {
        if inner.sources.get(&request.group) != Some(&request.group) {
            return Err(StateError::Invalid("writes require a known owned group"));
        }
        let key = (request.group.clone(), request.rank);
        let pool = inner
            .pools
            .get(&key)
            .ok_or(StateError::Invalid("unknown owner rank"))?;
        let count = request.blocks as usize;
        if count == 0 || plan.insert(key, count).is_some() {
            return Err(StateError::Invalid("zero or duplicate allocation request"));
        }
        if count > pool.blocks.num_free_blocks() {
            return Err(StateError::Invalid("pool capacity is not sufficient"));
        }
        if pool.next_generation == u64::MAX {
            return Err(StateError::Invalid("slot generation exhausted"));
        }
    }
    Ok(plan)
}

fn validate_reads(inner: &Inner, blocks: &[StateBlock]) -> Result<(), StateError> {
    for block in blocks {
        let pool = inner
            .pools
            .get(&(block.group.clone(), block.rank))
            .ok_or(StateError::Invalid("unknown read pool"))?;
        if block.slot == 0
            || block.slot >= pool.blocks.num_blocks()
            || pool.generations[block.slot as usize] != block.generation
            || pool.blocks.block(block.slot).ref_cnt == 0
            || pool.blocks.block(block.slot).ref_cnt == u32::MAX
        {
            return Err(StateError::Invalid(
                "stale read slot or reference count overflow",
            ));
        }
    }
    Ok(())
}

fn pin_blocks(inner: &mut Inner, blocks: &[StateBlock]) {
    for block in blocks {
        inner
            .pools
            .get_mut(&(block.group.clone(), block.rank))
            .expect("validated read pool")
            .blocks
            .touch(&[block.slot]);
    }
}

impl StorageLease {
    pub fn blocks(&self) -> &[StateBlock] {
        &self.blocks
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn pin(&self) -> Result<Self, StateError> {
        let mut inner = lock(&self.inner)?;
        validate_reads(&inner, &self.blocks)?;
        pin_blocks(&mut inner, &self.blocks);
        Ok(Self {
            inner: Arc::clone(&self.inner),
            epoch: self.epoch,
            blocks: self.blocks.clone(),
        })
    }
}

impl Drop for StorageLease {
    fn drop(&mut self) {
        // A poisoned arena cannot allocate again; do not recycle possibly inconsistent slots.
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        for block in &self.blocks {
            inner
                .pools
                .get_mut(&(block.group.clone(), block.rank))
                .expect("lease owns its pool")
                .blocks
                .free_blocks(&[block.slot]);
        }
    }
}

impl StateTransaction {
    pub fn key(&self) -> TransactionKey {
        self.key
    }

    pub fn blocks(&self) -> &[StateBlock] {
        self.writes.blocks()
    }

    pub fn read_blocks(&self) -> &[StateBlock] {
        self.reads.blocks()
    }

    pub fn abort(self) {
        drop(self);
    }

    pub fn into_storage(self) -> StorageLease {
        let Self { writes, reads, .. } = self;
        drop(reads);
        writes
    }
}

#[cfg(test)]
mod tests {
    use super::pool_capacity;

    #[test]
    fn capacities_leave_space_for_queue_sentinels() {
        assert_eq!(pool_capacity(2).unwrap(), 2);
        assert_eq!(
            pool_capacity(u64::from(u32::MAX - 2)).unwrap(),
            u32::MAX - 2
        );
        for capacity in [0, 1, u64::from(u32::MAX - 1), u64::from(u32::MAX), u64::MAX] {
            assert!(pool_capacity(capacity).is_err(), "capacity {capacity}");
        }
    }
}
