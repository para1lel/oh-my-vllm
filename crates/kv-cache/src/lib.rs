//! KV cache block management for oh-my-vllm.
//!
//! A port of the parts of vLLM's `vllm/v1/core/` that dominate scheduler CPU time:
//! the free-block LRU queue, block allocation with reference counting, and the
//! chain-hashed prefix cache index.
//!
//! The layering matches vLLM:
//!
//! - [`block`] — [`KVCacheBlock`] and [`FreeKVCacheBlockQueue`], the intrusive
//!   doubly-linked LRU list. vLLM hand-rolls this in Python to dodge object
//!   allocation; here it is an index arena over one contiguous `Vec`.
//! - [`hash`] — chain hashing. Each block hash folds in its parent's, so a single
//!   hash-map lookup per block resolves a whole prefix. This is why vLLM needs no
//!   radix tree, and it is the performance-SOTA arrangement we are matching.
//! - [`pool`] — [`BlockPool`], the only owner of `ref_cnt` and of the
//!   hash → block index. Implements vLLM's two-way eviction ordering: uncached
//!   blocks are reused LIFO (cache-hot), cached blocks FIFO (LRU).
//!
//! Nothing here touches CUDA memory. Block ids are indices into a KV cache tensor
//! that the Python worker owns; this crate only decides which ids go where.

pub mod block;
pub mod coordinator;
pub mod group;
pub mod hash;
pub mod pool;

pub use block::{FreeKVCacheBlockQueue, KVCacheBlock};
pub use coordinator::HybridCoordinator;
pub use group::{BlocksNeeded, CacheRequest, GroupKind, GroupManager, RequestId};
pub use hash::{BlockHash, BlockHashWithGroupId, hash_block_tokens, hash_request_tokens};
pub use pool::{BlockPool, NULL_BLOCK_ID};

#[cfg(test)]
mod tests {
    mod coordinator_tests;
}
