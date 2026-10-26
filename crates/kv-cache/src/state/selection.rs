use std::collections::BTreeSet;
use std::sync::Arc;

use super::{
    StateBlock, StateError, StateTransaction, StorageLease, lock, pin_blocks, validate_reads,
};

fn selected_storage<'a>(
    source: &StorageLease,
    authorized: impl Iterator<Item = &'a StateBlock>,
    blocks: &[StateBlock],
) -> Result<StorageLease, StateError> {
    let owned: BTreeSet<_> = authorized.collect();
    let selected: BTreeSet<_> = blocks.iter().collect();
    if selected.len() != blocks.len() || !selected.is_subset(&owned) {
        return Err(StateError::Invalid(
            "selection requires unique blocks held by the source owner",
        ));
    }
    let mut inner = lock(&source.inner)?;
    validate_reads(&inner, blocks)?;
    pin_blocks(&mut inner, blocks);
    Ok(StorageLease {
        inner: Arc::clone(&source.inner),
        epoch: source.epoch,
        blocks: blocks.to_vec(),
    })
}

impl StorageLease {
    pub fn select(&self, blocks: &[StateBlock]) -> Result<Self, StateError> {
        selected_storage(self, self.blocks.iter(), blocks)
    }
}

impl StateTransaction {
    pub fn select_storage(&self, blocks: &[StateBlock]) -> Result<StorageLease, StateError> {
        selected_storage(
            &self.writes,
            self.writes.blocks.iter().chain(&self.reads.blocks),
            blocks,
        )
    }
}
