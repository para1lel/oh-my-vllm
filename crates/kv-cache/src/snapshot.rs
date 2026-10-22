use rustc_hash::{FxHashMap, FxHashSet};

use crate::{HybridCoordinator, group::RequestId};

#[derive(Clone, Debug)]
pub struct BlockTables {
    pub fa: Vec<u32>,
    pub mamba: Vec<u32>,
    pub mamba_last_state: Option<usize>,
    pub mamba_read_source: Option<u32>,
}

impl BlockTables {
    pub fn unique(blocks: &[u32]) -> Vec<u32> {
        let mut seen = FxHashSet::default();
        blocks
            .iter()
            .copied()
            .filter(|&id| id != 0 && seen.insert(id))
            .collect()
    }

    pub fn fa_sources(&self) -> Vec<u32> {
        Self::unique(&self.fa)
    }

    pub fn mamba_sources(&self) -> Vec<u32> {
        let mut sources = self.mamba.clone();
        sources.extend(self.mamba_read_source);
        Self::unique(&sources)
    }
}

impl HybridCoordinator {
    pub fn request_tables(&self, request_id: RequestId) -> BlockTables {
        BlockTables {
            fa: self.full_attn_blocks(request_id).to_vec(),
            mamba: self.mamba_blocks(request_id).to_vec(),
            mamba_last_state: self.mamba.last_state_index(request_id),
            mamba_read_source: self.prefix_state_slot(request_id),
        }
    }

    pub fn pin_tables(&mut self, tables: &BlockTables) {
        self.pool.touch(&tables.fa_sources());
        self.mamba_pool
            .as_mut()
            .unwrap_or(&mut self.pool)
            .touch(&tables.mamba_sources());
    }

    pub fn release_tables(&mut self, tables: &BlockTables) {
        self.pool.free_blocks(&tables.fa_sources());
        self.mamba_pool
            .as_mut()
            .unwrap_or(&mut self.pool)
            .free_blocks(&tables.mamba_sources());
    }

    pub fn reserve_tables(&mut self, source: &BlockTables) -> Option<BlockTables> {
        let fa = source.fa_sources();
        let mb = source.mamba_sources();
        let fits = match &self.mamba_pool {
            Some(pool) => {
                fa.len() <= self.pool.num_free_blocks() && mb.len() <= pool.num_free_blocks()
            }
            None => fa.len() + mb.len() <= self.pool.num_free_blocks(),
        };
        if !fits {
            return None;
        }
        let fa_new = self
            .pool
            .get_new_blocks(fa.len())
            .expect("capacity checked");
        let mb_new = self
            .mamba_pool
            .as_mut()
            .unwrap_or(&mut self.pool)
            .get_new_blocks(mb.len())
            .expect("capacity checked");

        fn remap(table: &[u32], mapping: &FxHashMap<u32, u32>) -> Vec<u32> {
            table
                .iter()
                .map(|id| {
                    if *id == 0 {
                        0
                    } else {
                        mapping[id]
                    }
                })
                .collect()
        }

        let fa_mapping = fa.into_iter().zip(fa_new).collect();
        let mb_mapping: FxHashMap<_, _> = mb.into_iter().zip(mb_new).collect();
        Some(BlockTables {
            fa: remap(&source.fa, &fa_mapping),
            mamba: remap(&source.mamba, &mb_mapping),
            mamba_last_state: source.mamba_last_state,
            mamba_read_source: source.mamba_read_source.map(|id| mb_mapping[&id]),
        })
    }

    pub fn install_tables(
        &mut self,
        request_id: RequestId,
        tables: BlockTables,
        prefix: bool,
        computed: usize,
    ) {
        let read_source = tables.mamba_read_source.or_else(|| {
            (prefix && computed > 0).then(|| tables.mamba[(computed - 1) / self.block_size()])
        });
        if let Some(slot) = read_source {
            assert_ne!(slot, 0);
            self.restores.insert(request_id, slot);
        }
        self.full_attn
            .restore_blocks(request_id, tables.fa, !prefix, None, None);
        self.mamba.restore_blocks(
            request_id,
            tables.mamba,
            !prefix,
            tables.mamba_last_state,
            read_source,
        );
    }
}
