use super::*;
use oh_my_vllm_kv_cache::state::{RowVersion, StateBlock, StateTransaction};

#[test]
fn append_rows_require_live_pins_and_publish_atomic_versions() {
    let mut scenario = RowScenario::new();
    let (published, contender) = scenario.publish_initial();
    let (newest, retry) = scenario.validate_versions(&published, contender);
    scenario.release_and_reuse(published, newest, retry);
}

struct RowScenario {
    epoch: u64,
    owner: u64,
    state: StateArena,
    initial: StateTransaction,
    blocks: Vec<StateBlock>,
    first: RowVersion,
    second: RowVersion,
    foreign_tx: StateTransaction,
    unpinned: StateTransaction,
    other_request: StateTransaction,
}

impl RowScenario {
    fn new() -> Self {
        let epoch = (1_u64 << 40) + 9;
        let owner = (1_u64 << 40) + 41;
        let state = arena(epoch);
        let mut initial = state
            .reserve(1, owner, &[request("latent", 0, 2)], &[])
            .unwrap();
        let blocks = initial.blocks().to_vec();
        assert!(initial.create_row_root(&blocks[0], 0).is_err());
        let first = initial.create_row_root(&blocks[0], 8).unwrap();
        let second = initial.create_row_root(&blocks[1], 4).unwrap();
        assert!(initial.create_row_root(&blocks[0], 9).is_err());
        assert_eq!(first.block(), &blocks[0]);
        assert_eq!(first.storage().blocks(), &blocks[..1]);
        assert_eq!((first.version(), first.rows(), first.capacity()), (0, 0, 8));

        let foreign = arena(epoch);
        let foreign_tx = foreign
            .reserve(1, owner, &[request("latent", 0, 2)], &[])
            .unwrap();
        let foreign_root = foreign_tx
            .create_row_root(&foreign_tx.blocks()[0], 8)
            .unwrap();
        assert_eq!(foreign_root.block(), first.block());
        assert!(initial.append_rows(&[(foreign_root, 1)]).is_err());
        let mut forged = blocks[0].clone();
        forged.generation += 1;
        assert!(initial.create_row_root(&forged, 8).is_err());

        let mut unpinned = state.reserve(2, owner, &[], &[]).unwrap();
        assert!(unpinned.append_rows(&[(first.clone(), 1)]).is_err());
        let mut other_request = state
            .reserve(2, owner + (1 << 32), &[], &[first.storage()])
            .unwrap();
        initial.validate_row_version(&first).unwrap();
        other_request.validate_row_version(&first).unwrap();
        assert!(unpinned.validate_row_version(&first).is_err());
        assert!(foreign_tx.validate_row_version(&first).is_err());
        assert!(other_request.append_rows(&[(first.clone(), 1)]).is_err());
        assert!(other_request.create_row_root(first.block(), 8).is_err());

        Self {
            epoch,
            owner,
            state,
            initial,
            blocks,
            first,
            second,
            foreign_tx,
            unpinned,
            other_request,
        }
    }

    fn publish_initial(&mut self) -> (Vec<RowVersion>, StateTransaction) {
        assert!(self.initial.append_rows(&[]).is_ok());
        assert!(self.initial.row_grants().is_empty());
        assert!(self.initial.publish_rows(&[]).is_err());
        assert!(
            self.initial
                .append_rows(&[(self.first.clone(), 0)])
                .is_err()
        );
        assert!(
            self.initial
                .append_rows(&[(self.first.clone(), 1), (self.first.clone(), 1)])
                .is_err()
        );
        assert!(
            self.initial
                .append_rows(&[(self.first.clone(), 4), (self.second.clone(), 5)])
                .is_err()
        );
        assert!(self.initial.row_grants().is_empty());
        self.initial
            .append_rows(&[(self.first.clone(), 4), (self.second.clone(), 2)])
            .unwrap();
        let grants = self.initial.row_grants().to_vec();
        assert_eq!(grants.len(), 2);
        for (grant, block, end) in [
            (&grants[0], &self.blocks[0], 4),
            (&grants[1], &self.blocks[1], 2),
        ] {
            assert_eq!(grant.transaction, self.initial.key());
            assert_eq!(&grant.root, block);
            assert_eq!((grant.version, grant.begin, grant.end), (0, 0, end));
        }
        assert!(
            self.initial
                .append_rows(&[(self.first.clone(), 1)])
                .is_err()
        );
        let mut contender = self
            .state
            .reserve(
                3,
                self.owner,
                &[],
                &[self.first.storage(), self.second.storage()],
            )
            .unwrap();
        assert!(contender.append_rows(&[(self.second.clone(), 1)]).is_err());
        assert!(self.initial.publish_rows(&[3]).is_err());
        assert!(self.initial.publish_rows(&[3, 3]).is_err());
        assert!(contender.append_rows(&[(self.first.clone(), 1)]).is_err());
        let published = self.initial.publish_rows(&[3, 0]).unwrap();
        assert!(self.initial.row_grants().is_empty());
        assert_eq!(published.len(), 2);
        assert_eq!((published[0].version(), published[0].rows()), (1, 3));
        assert_eq!((published[1].version(), published[1].rows()), (1, 0));
        assert_eq!((self.first.version(), self.first.rows()), (0, 0));
        (published, contender)
    }

    fn validate_versions(
        &mut self,
        published: &[RowVersion],
        mut contender: StateTransaction,
    ) -> (RowVersion, StateTransaction) {
        self.initial.validate_row_read(&published[0], 0, 3).unwrap();
        self.other_request
            .validate_row_read(&published[0], 1, 3)
            .unwrap();
        assert!(
            self.unpinned
                .validate_row_read(&published[0], 0, 1)
                .is_err()
        );
        assert!(
            self.foreign_tx
                .validate_row_read(&published[0], 0, 1)
                .is_err()
        );
        assert!(self.initial.validate_row_read(&self.first, 0, 1).is_err());
        for (begin, end) in [(0, 0), (2, 1), (0, 4), (3, u32::MAX)] {
            assert!(
                self.initial
                    .validate_row_read(&published[0], begin, end)
                    .is_err()
            );
        }
        assert!(self.initial.publish_rows(&[3, 0]).is_err());
        assert!(
            self.initial
                .append_rows(&[(published[0].clone(), 1)])
                .is_err()
        );
        assert!(contender.append_rows(&[(self.first.clone(), 1)]).is_err());
        assert!(contender.append_rows(&[(published[0].clone(), 6)]).is_err());
        contender.append_rows(&[(published[0].clone(), 2)]).unwrap();
        self.other_request
            .validate_row_read(&published[0], 0, 3)
            .unwrap();
        assert!(contender.validate_row_read(&published[0], 0, 5).is_err());
        assert_eq!(
            (
                contender.row_grants()[0].begin,
                contender.row_grants()[0].end
            ),
            (3, 5)
        );
        assert!(contender.publish_rows(&[2]).is_err());
        let detached = vec![contender];
        let mut retry = self
            .state
            .reserve(4, self.owner, &[], &[published[0].storage()])
            .unwrap();
        assert!(retry.append_rows(&[(published[0].clone(), 1)]).is_err());
        drop(detached);
        retry.append_rows(&[(published[0].clone(), 1)]).unwrap();
        assert_eq!(
            (retry.row_grants()[0].begin, retry.row_grants()[0].end),
            (3, 4)
        );
        let newest = retry.publish_rows(&[4]).unwrap().remove(0);
        assert_eq!((newest.version(), newest.rows()), (2, 4));
        self.other_request
            .validate_row_read(&published[0], 0, 3)
            .unwrap();
        assert!(
            self.other_request
                .validate_row_read(&published[0], 0, 4)
                .is_err()
        );
        assert!(
            self.other_request
                .append_rows(&[(newest.clone(), 1)])
                .is_err()
        );
        (newest, retry)
    }

    fn release_and_reuse(
        self,
        published: Vec<RowVersion>,
        newest: RowVersion,
        retry: StateTransaction,
    ) {
        let Self {
            epoch,
            owner,
            state,
            initial,
            blocks,
            first,
            second,
            unpinned,
            other_request,
            foreign_tx: _foreign_tx,
        } = self;
        drop(retry);
        drop(initial.into_storage());
        drop(unpinned);
        drop(other_request);
        drop(published);
        drop(second);
        drop(newest);
        assert_eq!(state.free_blocks("latent", 0).unwrap(), 2);
        drop(first);
        assert_eq!(state.free_blocks("latent", 0).unwrap(), 3);

        let mut reused = state
            .reserve(5, owner, &[request("latent", 0, 3)], &[])
            .unwrap();
        let reused_block = reused
            .blocks()
            .iter()
            .find(|block| block.slot == blocks[0].slot)
            .unwrap()
            .clone();
        assert!(reused_block.generation > blocks[0].generation);
        let root = reused.create_row_root(&reused_block, u32::MAX).unwrap();
        assert_eq!((root.version(), root.rows()), (0, 0));
        assert!(reused.create_row_root(&blocks[0], 8).is_err());
        reused.append_rows(&[(root.clone(), u32::MAX)]).unwrap();
        let full = reused.publish_rows(&[u32::MAX]).unwrap().remove(0);
        let mut overflow = state.reserve(6, owner, &[], &[full.storage()]).unwrap();
        assert!(overflow.append_rows(&[(full, 1)]).is_err());
        assert!(overflow.row_grants().is_empty());
        drop(overflow);
        drop(state);
        drop(reused.into_storage());
        assert_eq!(root.storage().epoch(), epoch);
        assert_eq!(root.block(), &reused_block);
    }
}
