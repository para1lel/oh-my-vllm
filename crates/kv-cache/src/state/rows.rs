use std::collections::BTreeSet;
use std::fmt;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::{
    Inner, PoolKey, StateBlock, StateError, StateTransaction, StorageLease, TransactionKey, lock,
    pin_blocks, validate_reads,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowGrant {
    pub transaction: TransactionKey,
    pub root: StateBlock,
    pub version: u64,
    pub begin: u32,
    pub end: u32,
}

struct RowOwner {
    storage: StorageLease,
    capacity: u32,
}

#[derive(Clone)]
pub struct RowVersion {
    owner: Arc<RowOwner>,
    version: u64,
    end: u32,
}

impl fmt::Debug for RowVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RowVersion")
            .field("block", self.block())
            .field("version", &self.version)
            .field("rows", &self.end)
            .field("capacity", &self.capacity())
            .finish()
    }
}

impl RowVersion {
    pub fn block(&self) -> &StateBlock {
        &self.owner.storage.blocks[0]
    }

    pub fn storage(&self) -> &StorageLease {
        &self.owner.storage
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn rows(&self) -> u32 {
        self.end
    }

    pub fn capacity(&self) -> u32 {
        self.owner.capacity
    }
}

pub(super) struct RowRoot {
    generation: u64,
    request_id: u64,
    capacity: u32,
    version: u64,
    end: u32,
    active: Option<TransactionKey>,
}

fn root_key(block: &StateBlock) -> (PoolKey, u32) {
    ((block.group.clone(), block.rank), block.slot)
}

pub(super) struct AppendGuard {
    inner: Arc<Mutex<Inner>>,
    transaction: TransactionKey,
    versions: Vec<RowVersion>,
    active: bool,
}

impl Drop for AppendGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        for version in &self.versions {
            if let Some(root) = inner.row_roots.get_mut(&root_key(version.block()))
                && root.generation == version.block().generation
                && root.active == Some(self.transaction)
            {
                root.active = None;
            }
        }
    }
}

impl StateTransaction {
    pub fn create_row_root(
        &self,
        block: &StateBlock,
        capacity: u32,
    ) -> Result<RowVersion, StateError> {
        if capacity == 0 || !self.writes.blocks.contains(block) {
            return Err(StateError::Invalid(
                "row root requires a positive owned write allocation",
            ));
        }
        let mut inner = lock(&self.writes.inner)?;
        validate_reads(&inner, std::slice::from_ref(block))?;
        let key = root_key(block);
        if inner
            .row_roots
            .get(&key)
            .is_some_and(|root| root.generation == block.generation)
        {
            return Err(StateError::Invalid(
                "row root generation is already registered",
            ));
        }
        pin_blocks(&mut inner, std::slice::from_ref(block));
        inner.row_roots.insert(
            key,
            RowRoot {
                generation: block.generation,
                request_id: self.key.request_id,
                capacity,
                version: 0,
                end: 0,
                active: None,
            },
        );
        Ok(RowVersion {
            owner: Arc::new(RowOwner {
                storage: StorageLease {
                    inner: Arc::clone(&self.writes.inner),
                    epoch: self.key.epoch,
                    blocks: vec![block.clone()],
                },
                capacity,
            }),
            version: 0,
            end: 0,
        })
    }

    fn current_root<'a>(
        &self,
        inner: &'a Inner,
        version: &RowVersion,
    ) -> Result<&'a RowRoot, StateError> {
        if !Arc::ptr_eq(&self.writes.inner, &version.storage().inner)
            || version.storage().epoch != self.key.epoch
            || (!self.writes.blocks.contains(version.block())
                && !self.reads.blocks.contains(version.block()))
        {
            return Err(StateError::Invalid(
                "row source requires this transaction's live pin",
            ));
        }
        let root = inner
            .row_roots
            .get(&root_key(version.block()))
            .ok_or(StateError::Invalid("unknown row root"))?;
        if root.generation != version.block().generation
            || root.request_id != self.key.request_id
            || root.capacity != version.capacity()
            || root.version != version.version
            || root.end != version.end
        {
            return Err(StateError::Invalid("foreign request or stale row version"));
        }
        Ok(root)
    }

    pub fn validate_row_version(&self, version: &RowVersion) -> Result<(), StateError> {
        if !Arc::ptr_eq(&self.writes.inner, &version.storage().inner)
            || version.storage().epoch != self.key.epoch
            || (!self.writes.blocks.contains(version.block())
                && !self.reads.blocks.contains(version.block()))
        {
            return Err(StateError::Invalid(
                "row snapshot requires this transaction's live pin",
            ));
        }
        let inner = lock(&self.writes.inner)?;
        let root = inner
            .row_roots
            .get(&root_key(version.block()))
            .ok_or(StateError::Invalid("unknown row root"))?;
        // Published snapshots remain readable across appends and prefix sharing.
        if root.generation != version.block().generation
            || root.capacity != version.capacity()
            || root.version < version.version()
            || root.end < version.rows()
        {
            return Err(StateError::Invalid("row read has an invalid root version"));
        }
        Ok(())
    }

    pub fn validate_row_read(
        &self,
        version: &RowVersion,
        begin: u32,
        end: u32,
    ) -> Result<(), StateError> {
        if begin >= end || end > version.rows() {
            return Err(StateError::Invalid(
                "row read exceeds the pinned published version",
            ));
        }
        self.validate_row_version(version)
    }

    pub fn append_rows(&mut self, requests: &[(RowVersion, u32)]) -> Result<(), StateError> {
        if requests.is_empty() {
            return Ok(());
        }
        if self.row_issued {
            return Err(StateError::Invalid("row append batch already issued"));
        }
        let mut inner = lock(&self.writes.inner)?;
        let mut roots = BTreeSet::new();
        let mut grants = Vec::with_capacity(requests.len());
        for (version, count) in requests {
            if !roots.insert(root_key(version.block())) || *count == 0 {
                return Err(StateError::Invalid(
                    "duplicate row root or zero append count",
                ));
            }
            let root = self.current_root(&inner, version)?;
            if root.active.is_some() {
                return Err(StateError::Invalid("row root already has an active append"));
            }
            let end = version
                .end
                .checked_add(*count)
                .filter(|end| *end <= root.capacity)
                .ok_or(StateError::Invalid("row append exceeds capacity"))?;
            grants.push(RowGrant {
                transaction: self.key,
                root: version.block().clone(),
                version: version.version,
                begin: version.end,
                end,
            });
        }
        let guard = AppendGuard {
            inner: Arc::clone(&self.writes.inner),
            transaction: self.key,
            versions: requests
                .iter()
                .map(|(version, _)| version.clone())
                .collect(),
            active: true,
        };
        for key in roots {
            inner
                .row_roots
                .get_mut(&key)
                .expect("validated root")
                .active = Some(self.key);
        }
        self.row_append = Some(guard);
        self.row_grants = grants;
        self.row_issued = true;
        Ok(())
    }

    pub fn row_grants(&self) -> &[RowGrant] {
        &self.row_grants
    }

    pub fn publish_rows(&mut self, accepted_ends: &[u32]) -> Result<Vec<RowVersion>, StateError> {
        let guard = self
            .row_append
            .as_ref()
            .filter(|guard| guard.active)
            .ok_or(StateError::Invalid("no unpublished row append"))?;
        if accepted_ends.len() != self.row_grants.len() {
            return Err(StateError::Invalid(
                "accepted row counts do not match grants",
            ));
        }
        let mut inner = lock(&self.writes.inner)?;
        let mut published = Vec::with_capacity(accepted_ends.len());
        for ((source, grant), &end) in guard
            .versions
            .iter()
            .zip(&self.row_grants)
            .zip(accepted_ends)
        {
            let root = self.current_root(&inner, source)?;
            if root.active != Some(self.key) || !(grant.begin..=grant.end).contains(&end) {
                return Err(StateError::Invalid(
                    "invalid active append or accepted row end",
                ));
            }
            let version = root
                .version
                .checked_add(1)
                .ok_or(StateError::Invalid("row version exhausted"))?;
            published.push(RowVersion {
                owner: Arc::clone(&source.owner),
                version,
                end,
            });
        }
        for version in &published {
            let root = inner
                .row_roots
                .get_mut(&root_key(version.block()))
                .expect("validated root");
            root.version = version.version;
            root.end = version.end;
            root.active = None;
        }
        self.row_append.as_mut().expect("validated guard").active = false;
        self.row_grants.clear();
        Ok(published)
    }
}
