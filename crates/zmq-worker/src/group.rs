use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use oh_my_vllm_model_contract::{
    CacheSchema, CacheSource, ContractError, ModelDescriptor, RequestPlacement,
};

pub mod protocol;

use protocol::{
    ExecutionMode, GROUP_PROTOCOL_VERSION, RankHeader, RankReady, RankResult, RankSchedule,
    StepDispatch, StepPlan,
};

#[derive(Debug, thiserror::Error)]
pub enum GroupError {
    #[error(transparent)]
    Contract(#[from] ContractError),
    #[error("invalid rank group: {0}")]
    Invalid(&'static str),
    #[error("rank group has failed")]
    Failed,
}

type Result<T> = std::result::Result<T, GroupError>;

pub struct GroupLimits {
    pub max_slots: u64,
    pub max_requests: usize,
    pub max_rows: u32,
}

pub struct GroupCapacities {
    pub common_schema: CacheSchema,
    pub per_rank: BTreeMap<u32, BTreeMap<String, u64>>,
    pub owned_slots: u64,
}

pub struct CompletedStep {
    epoch: u64,
    step_id: u64,
    transactions: Vec<(u64, u64)>,
}

#[derive(PartialEq, Eq)]
enum Phase {
    Starting,
    Ready,
    Running,
    Failed,
}

struct ActiveStep {
    dispatch: StepDispatch,
    transactions: Vec<(u64, u64)>,
    completed: BTreeSet<u32>,
}

pub struct GroupGate {
    epoch: u64,
    expected: ModelDescriptor,
    devices: Vec<String>,
    limits: GroupLimits,
    phase: Phase,
    deadline: Option<Instant>,
    capacities: GroupCapacities,
    last_step: u64,
    last_transaction: u64,
    active: Option<ActiveStep>,
}

fn require(condition: bool, reason: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(GroupError::Invalid(reason))
    }
}

fn valid_device(value: &str) -> bool {
    value.len() == 40
        && value.starts_with("GPU-")
        && value.as_bytes()[4..].iter().enumerate().all(|(i, &byte)| {
            if [8, 13, 18, 23].contains(&i) {
                byte == b'-'
            } else {
                byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
            }
        })
}

fn owned_slots(schema: &CacheSchema) -> Result<u64> {
    let mut total = 0_u64;
    for group in &schema.groups {
        require(
            (2..=u64::from(u32::MAX - 2)).contains(&group.capacity_blocks),
            "cache capacity does not leave space for pool sentinels",
        )?;
        match &group.source {
            CacheSource::Owned {} => {
                total = total
                    .checked_add(group.capacity_blocks)
                    .ok_or(GroupError::Invalid("capacity sum overflow"))?;
            }
            CacheSource::Shared { producer } => {
                let source = schema.groups.iter().find(|source| &source.name == producer);
                require(
                    source.is_some_and(|source| {
                        source.kind == group.kind
                            && source.block_tokens == group.block_tokens
                            && source.storage == group.storage
                            && source.capacity_blocks == group.capacity_blocks
                    }),
                    "shared storage geometry differs from its producer",
                )?;
            }
        }
    }
    Ok(total)
}

impl GroupGate {
    pub fn new(
        epoch: u64,
        expected: ModelDescriptor,
        devices: Vec<String>,
        limits: GroupLimits,
        startup_deadline: Instant,
    ) -> Result<Self> {
        expected.validate()?;
        require(epoch > 0, "epoch must be positive")?;
        require(
            limits.max_requests > 0 && limits.max_rows > 0,
            "request and row limits must be positive",
        )?;
        require(
            devices.len() == expected.parallel_plan.world_size as usize
                && devices.iter().all(|device| valid_device(device))
                && devices.iter().collect::<BTreeSet<_>>().len() == devices.len(),
            "expected one distinct GPU UUID per logical rank",
        )?;
        let slots = owned_slots(&expected.cache_schema)?;
        require(
            slots
                .checked_mul(u64::from(expected.parallel_plan.world_size))
                .is_some_and(|slots| slots <= limits.max_slots),
            "minimum capacity exceeds logical slot budget",
        )?;
        let capacities = GroupCapacities {
            common_schema: expected.cache_schema.clone(),
            per_rank: BTreeMap::new(),
            owned_slots: 0,
        };
        Ok(Self {
            epoch,
            expected,
            devices,
            limits,
            phase: Phase::Starting,
            deadline: Some(startup_deadline),
            capacities,
            last_step: 0,
            last_transaction: 0,
            active: None,
        })
    }

    pub fn is_failed(&self) -> bool {
        self.phase == Phase::Failed
    }

    pub fn capacities(&self) -> Option<&GroupCapacities> {
        matches!(self.phase, Phase::Ready | Phase::Running).then_some(&self.capacities)
    }

    pub fn fail_rank(&mut self, _rank: u32) {
        self.phase = Phase::Failed;
        self.active = None;
        self.deadline = None;
    }

    pub fn check_deadline(&mut self, now: Instant) -> Result<()> {
        if self.is_failed() {
            return Err(GroupError::Failed);
        }
        if self.deadline.is_some_and(|deadline| now >= deadline) {
            self.fail_rank(0);
            return Err(GroupError::Invalid("group deadline expired"));
        }
        Ok(())
    }

    fn header(&self, source_rank: u32, header: &RankHeader) -> Result<()> {
        require(
            header.version == GROUP_PROTOCOL_VERSION
                && header.epoch == self.epoch
                && header.rank == source_rank
                && source_rank < self.expected.parallel_plan.world_size,
            "rank header differs from its bound connection",
        )
    }

    pub fn accept_ready(
        &mut self,
        source_rank: u32,
        ready: RankReady,
        now: Instant,
    ) -> Result<bool> {
        self.check_deadline(now)?;
        let result = self.insert_ready(source_rank, ready);
        if result.is_err() {
            self.fail_rank(source_rank);
        }
        result
    }

    fn insert_ready(&mut self, source_rank: u32, ready: RankReady) -> Result<bool> {
        require(self.phase == Phase::Starting, "unexpected rank readiness")?;
        self.header(source_rank, &ready.header)?;
        require(
            ready.device_uuid == self.devices[source_rank as usize]
                && !self.capacities.per_rank.contains_key(&source_rank),
            "wrong device or duplicate readiness",
        )?;
        ready.descriptor.validate()?;
        let slots = owned_slots(&ready.descriptor.cache_schema)?;
        let mut normalized = ready.descriptor.clone();
        require(
            normalized.cache_schema.groups.len() == self.expected.cache_schema.groups.len(),
            "different cache group count",
        )?;
        for (actual, expected) in normalized
            .cache_schema
            .groups
            .iter_mut()
            .zip(&self.expected.cache_schema.groups)
        {
            require(
                actual.capacity_blocks >= expected.capacity_blocks,
                "rank capacity is less than the required minimum",
            )?;
            actual.capacity_blocks = expected.capacity_blocks;
        }
        require(
            normalized == self.expected,
            "rank descriptor differs from group",
        )?;
        let total = self
            .capacities
            .owned_slots
            .checked_add(slots)
            .filter(|&slots| slots <= self.limits.max_slots)
            .ok_or(GroupError::Invalid(
                "rank capacity exceeds logical slot budget",
            ))?;
        self.capacities.per_rank.insert(
            source_rank,
            ready
                .descriptor
                .cache_schema
                .groups
                .iter()
                .map(|group| (group.name.clone(), group.capacity_blocks))
                .collect(),
        );
        self.capacities.owned_slots = total;
        let complete = self.capacities.per_rank.len() == self.devices.len();
        if complete {
            for group in &mut self.capacities.common_schema.groups {
                group.capacity_blocks = self
                    .capacities
                    .per_rank
                    .values()
                    .map(|capacities| capacities[&group.name])
                    .min()
                    .expect("all ranks ready");
            }
            self.phase = Phase::Ready;
            self.deadline = None;
        }
        Ok(complete)
    }

    pub fn begin(
        &mut self,
        plan: StepPlan,
        deadline: Instant,
        now: Instant,
    ) -> Result<StepDispatch> {
        self.check_deadline(now)?;
        require(self.phase == Phase::Ready, "group is not ready for a step")?;
        require(deadline > now, "step deadline has already expired")?;
        let dispatch = self.dispatch(&plan)?;
        self.last_step = plan.step_id;
        if let Some(maximum) = plan
            .requests
            .iter()
            .map(|request| request.transaction_id)
            .max()
        {
            self.last_transaction = maximum;
        }
        self.active = Some(ActiveStep {
            dispatch: dispatch.clone(),
            transactions: plan
                .requests
                .iter()
                .map(|request| (request.request_id, request.transaction_id))
                .collect(),
            completed: BTreeSet::new(),
        });
        self.phase = Phase::Running;
        self.deadline = Some(deadline);
        Ok(dispatch)
    }

    fn dispatch(&self, plan: &StepPlan) -> Result<StepDispatch> {
        require(plan.step_id > self.last_step, "zero or stale step identity")?;
        require(
            plan.collective_plan_sha256.len() == 64
                && plan
                    .collective_plan_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "invalid collective plan digest",
        )?;
        require(
            plan.requests.len() <= self.limits.max_requests,
            "too many requests",
        )?;
        let replicated =
            self.expected.parallel_plan.request_placement == RequestPlacement::Replicated;
        let mut request_ids = BTreeSet::new();
        let mut transaction_ids = BTreeSet::new();
        let mut rows = 0_u32;
        for request in &plan.requests {
            require(
                request_ids.insert(request.request_id)
                    && transaction_ids.insert(request.transaction_id)
                    && request.transaction_id > self.last_transaction,
                "duplicate request or stale transaction identity",
            )?;
            require(
                request.owner_rank < self.expected.parallel_plan.world_size
                    && (!replicated || request.owner_rank == 0),
                "invalid request owner rank",
            )?;
            require(request.rows > 0, "request must have a positive row count")?;
            rows = rows
                .checked_add(request.rows)
                .filter(|&rows| rows <= self.limits.max_rows)
                .ok_or(GroupError::Invalid("step row limit exceeded"))?;
        }
        let ranks = (0..self.expected.parallel_plan.world_size)
            .map(|rank| {
                let requests: Vec<_> = plan
                    .requests
                    .iter()
                    .filter(|request| replicated || request.owner_rank == rank)
                    .cloned()
                    .collect();
                RankSchedule {
                    rank,
                    rows: requests.iter().map(|request| request.rows).sum(),
                    requests,
                }
            })
            .collect();
        Ok(StepDispatch {
            version: GROUP_PROTOCOL_VERSION,
            epoch: self.epoch,
            step_id: plan.step_id,
            mode: if plan.requests.iter().any(|request| request.prefill) {
                ExecutionMode::Eager
            } else {
                ExecutionMode::Graph
            },
            collective_plan_sha256: plan.collective_plan_sha256.clone(),
            collective_count: plan.collective_count,
            ranks,
        })
    }

    pub fn accept_result(
        &mut self,
        source_rank: u32,
        result: RankResult,
        now: Instant,
    ) -> Result<Option<CompletedStep>> {
        self.check_deadline(now)?;
        let completion = self.insert_result(source_rank, result);
        if completion.is_err() {
            self.fail_rank(source_rank);
        }
        completion
    }

    fn insert_result(
        &mut self,
        source_rank: u32,
        result: RankResult,
    ) -> Result<Option<CompletedStep>> {
        require(self.phase == Phase::Running, "unexpected rank result")?;
        self.header(source_rank, &result.header)?;
        let active = self.active.as_mut().expect("running step exists");
        require(
            result.step_id == active.dispatch.step_id
                && result.collective_plan_sha256 == active.dispatch.collective_plan_sha256
                && result.completed_collectives == active.dispatch.collective_count
                && result.device_work_complete
                && !active.completed.contains(&source_rank),
            "stale, duplicate, or incomplete rank result",
        )?;
        let expected: BTreeSet<_> = active.dispatch.ranks[source_rank as usize]
            .requests
            .iter()
            .map(|request| request.transaction_id)
            .collect();
        require(
            result.transaction_ids.len() == expected.len()
                && result
                    .transaction_ids
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
                    == expected,
            "rank transaction results differ from the schedule",
        )?;
        active.completed.insert(source_rank);
        if active.completed.len() != self.devices.len() {
            return Ok(None);
        }
        let active = self.active.take().expect("complete step exists");
        self.phase = Phase::Ready;
        self.deadline = None;
        Ok(Some(CompletedStep {
            epoch: self.epoch,
            step_id: active.dispatch.step_id,
            transactions: active.transactions,
        }))
    }
}

impl CompletedStep {
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn step_id(&self) -> u64 {
        self.step_id
    }

    pub fn transactions(&self) -> &[(u64, u64)] {
        &self.transactions
    }
}
