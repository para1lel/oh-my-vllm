use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{GroupError, Result, require};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectivePlan {
    pub version: u32,
    pub world_size: u32,
    pub operations: Vec<CollectiveOperation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum CollectiveOperation {
    AllReduce {
        dtype: CollectiveType,
        count: u64,
    },
    AllGather {
        dtype: CollectiveType,
        count: u64,
    },
    ReduceScatter {
        dtype: CollectiveType,
        count: u64,
    },
    Route {
        dtype: CollectiveType,
        counts: Vec<Vec<u64>>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CollectiveType {
    F32,
    Bf16,
    F16,
    I32,
    I64,
    U8,
}

impl CollectiveType {
    fn code(self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::Bf16 => 1,
            Self::F16 => 2,
            Self::I32 => 3,
            Self::I64 => 4,
            Self::U8 => 5,
        }
    }

    fn bytes(self) -> u64 {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::Bf16 | Self::F16 => 2,
            Self::I64 => 8,
            Self::U8 => 1,
        }
    }
}

impl CollectiveOperation {
    fn kind(&self) -> (u8, CollectiveType) {
        match self {
            Self::AllReduce { dtype, .. } => (0, *dtype),
            Self::AllGather { dtype, .. } => (1, *dtype),
            Self::ReduceScatter { dtype, .. } => (2, *dtype),
            Self::Route { dtype, .. } => (3, *dtype),
        }
    }

    fn counts(&self, world_size: u32, rank: u32) -> Result<(u64, u64)> {
        let scale = |count: u64| {
            count
                .checked_mul(u64::from(world_size))
                .ok_or(GroupError::Invalid("collective element count overflow"))
        };
        match self {
            Self::AllReduce { count, .. } => Ok((*count, *count)),
            Self::AllGather { count, .. } => Ok((*count, scale(*count)?)),
            Self::ReduceScatter { count, .. } => Ok((scale(*count)?, *count)),
            Self::Route { counts, .. } => {
                let rank = rank as usize;
                let sum = |total: u64, count: &u64| {
                    total
                        .checked_add(*count)
                        .ok_or(GroupError::Invalid("route element count overflow"))
                };
                Ok((
                    counts[rank].iter().try_fold(0, sum)?,
                    counts.iter().map(|row| &row[rank]).try_fold(0, sum)?,
                ))
            }
        }
    }
}

impl CollectivePlan {
    fn validate(&self, world_size: u32) -> Result<()> {
        require(
            self.version == 1 && matches!(self.world_size, 1 | 4) && self.world_size == world_size,
            "collective plan version or topology differs",
        )?;
        require(
            self.operations.len() <= 4096,
            "too many collective operations",
        )?;
        for operation in &self.operations {
            if let CollectiveOperation::Route { counts, .. } = operation {
                require(
                    counts.len() == world_size as usize
                        && counts.iter().all(|row| row.len() == world_size as usize),
                    "route counts must form a square world-size matrix",
                )?;
            }
            for rank in 0..world_size {
                let (input, output) = operation.counts(world_size, rank)?;
                for count in [input, output] {
                    require(
                        count
                            .checked_mul(operation.kind().1.bytes())
                            .is_some_and(|bytes| bytes <= i64::MAX as u64),
                        "collective buffer exceeds signed byte range",
                    )?;
                }
            }
        }
        Ok(())
    }

    pub fn identity(&self, world_size: u32) -> Result<String> {
        self.validate(world_size)?;
        let mut digest = Sha256::new();
        digest.update(b"oh-my-vllm/collectives/v1\0");
        for value in [self.version, self.world_size, self.operations.len() as u32] {
            digest.update(value.to_le_bytes());
        }
        for operation in &self.operations {
            let (op, dtype) = operation.kind();
            digest.update([op, dtype.code()]);
            match operation {
                CollectiveOperation::AllReduce { count, .. }
                | CollectiveOperation::AllGather { count, .. }
                | CollectiveOperation::ReduceScatter { count, .. } => {
                    digest.update(count.to_le_bytes());
                }
                CollectiveOperation::Route { counts, .. } => {
                    for row in counts {
                        for count in row {
                            digest.update(count.to_le_bytes());
                        }
                    }
                }
            }
        }
        Ok(format!("{:x}", digest.finalize()))
    }

    pub fn buffer_counts(&self, rank: u32) -> Result<Vec<(u64, u64)>> {
        self.validate(self.world_size)?;
        require(rank < self.world_size, "collective rank is out of range")?;
        self.operations
            .iter()
            .map(|operation| operation.counts(self.world_size, rank))
            .collect()
    }
}
