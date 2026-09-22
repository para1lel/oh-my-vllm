//! Chain hashing for the prefix cache.
//!
//! Mirrors vLLM's `kv_cache_utils.hash_block_tokens`: each block's hash folds in
//! the *parent* block's hash, so a hash identifies a full token prefix rather
//! than just the tokens inside one block. That is what makes a plain hash map
//! sufficient for prefix caching — no radix tree needed, because a single lookup
//! per block walks the prefix chain implicitly.

use sha2::{Digest, Sha256};

/// Hash of a token prefix ending at some block boundary.
///
/// Deviation from vLLM: vLLM keeps the full 32-byte SHA-256 digest, we truncate
/// to 64 bits so the map key stays `Copy` and 8 bytes wide. For a single-node
/// cache bounded by GPU memory (order 1e5 blocks) the collision probability is
/// negligible, and a collision degrades to serving a wrong cache hit exactly as
/// it would in vLLM under full-digest collision — the failure mode is unchanged,
/// only its probability.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BlockHash(pub u64);

/// A block hash qualified by KV cache group.
///
/// Hybrid models (Qwen3.8: GatedDeltaNet + full attention) have several KV cache
/// groups with different block layouts, so the same token prefix maps to a
/// different physical block per group. The group id keeps those entries distinct
/// in one shared map.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BlockHashWithGroupId {
    pub block_hash: BlockHash,
    pub group_id: u32,
}

impl BlockHashWithGroupId {
    #[inline]
    pub fn new(block_hash: BlockHash, group_id: u32) -> Self {
        Self {
            block_hash,
            group_id,
        }
    }
}

/// Extra cache-key material that must disambiguate otherwise identical tokens
/// (LoRA adapter id, multimodal item hashes, cache salt).
///
/// Text-only single-adapter serving leaves this empty; the type exists so the
/// hash is stable when multimodal support lands and starts populating it.
pub type ExtraKeys = [u64];

/// Hash one block's worth of tokens, chained onto the parent block's hash.
///
/// `parent_hash` is `None` for the first block of a request. Field lengths are
/// folded in alongside the fields themselves so that no two distinct
/// (parent, tokens, extra_keys) triples can produce the same byte stream.
pub fn hash_block_tokens(
    parent_hash: Option<BlockHash>,
    tokens: &[u32],
    extra_keys: &ExtraKeys,
) -> BlockHash {
    let mut hasher = Sha256::new();
    // Distinguish "no parent" from a parent that happens to hash to 0.
    match parent_hash {
        Some(p) => {
            hasher.update([1u8]);
            hasher.update(p.0.to_le_bytes());
        }
        None => hasher.update([0u8]),
    }
    hasher.update((tokens.len() as u64).to_le_bytes());
    for &t in tokens {
        hasher.update(t.to_le_bytes());
    }
    hasher.update((extra_keys.len() as u64).to_le_bytes());
    for &k in extra_keys {
        hasher.update(k.to_le_bytes());
    }
    let digest = hasher.finalize();
    BlockHash(u64::from_le_bytes(
        digest[..8].try_into().expect("sha256 digest is 32 bytes"),
    ))
}

/// Hash every full block of `token_ids`, chaining from `parent_hash`.
///
/// Trailing tokens that do not fill a block are ignored — a partial block has no
/// stable hash until it is filled (vLLM handles those separately via
/// `cache_partial_block`).
pub fn hash_request_tokens(
    parent_hash: Option<BlockHash>,
    token_ids: &[u32],
    block_size: usize,
    extra_keys: &ExtraKeys,
) -> Vec<BlockHash> {
    assert!(block_size > 0, "block_size must be positive");
    let mut hashes = Vec::with_capacity(token_ids.len() / block_size);
    let mut parent = parent_hash;
    for chunk in token_ids.chunks_exact(block_size) {
        let h = hash_block_tokens(parent, chunk, extra_keys);
        hashes.push(h);
        parent = Some(h);
    }
    hashes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic() {
        let a = hash_block_tokens(None, &[1, 2, 3], &[]);
        let b = hash_block_tokens(None, &[1, 2, 3], &[]);
        assert_eq!(a, b);
    }

    #[test]
    fn hash_depends_on_parent() {
        let root = hash_block_tokens(None, &[1, 2, 3], &[]);
        let child = hash_block_tokens(Some(root), &[4, 5, 6], &[]);
        let orphan = hash_block_tokens(None, &[4, 5, 6], &[]);
        assert_ne!(child, orphan, "chain hashing must fold in the parent");
    }

    #[test]
    fn hash_depends_on_extra_keys() {
        let plain = hash_block_tokens(None, &[1, 2, 3], &[]);
        let with_lora = hash_block_tokens(None, &[1, 2, 3], &[7]);
        assert_ne!(plain, with_lora);
    }

    #[test]
    fn length_prefixing_prevents_field_smearing() {
        // Without length prefixes these two would serialize identically.
        let a = hash_block_tokens(None, &[1, 2], &[3]);
        let b = hash_block_tokens(None, &[1, 2, 3], &[]);
        assert_ne!(a, b);
    }

    #[test]
    fn request_hashes_form_a_chain() {
        let tokens: Vec<u32> = (0..10).collect();
        let hashes = hash_request_tokens(None, &tokens, 4, &[]);
        // 10 tokens / block_size 4 => 2 full blocks, trailing 2 tokens dropped.
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes[0], hash_block_tokens(None, &[0, 1, 2, 3], &[]));
        assert_eq!(
            hashes[1],
            hash_block_tokens(Some(hashes[0]), &[4, 5, 6, 7], &[])
        );
    }

    #[test]
    fn shared_prefix_produces_shared_hashes() {
        let a: Vec<u32> = (0..12).collect();
        let mut b: Vec<u32> = (0..12).collect();
        b[9] = 999; // diverge inside the third block only

        let ha = hash_request_tokens(None, &a, 4, &[]);
        let hb = hash_request_tokens(None, &b, 4, &[]);
        assert_eq!(ha[..2], hb[..2], "common prefix blocks must match");
        assert_ne!(ha[2], hb[2]);
    }
}
