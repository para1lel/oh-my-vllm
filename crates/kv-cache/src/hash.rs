use sha2::{Digest, Sha256};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BlockHash(pub u64);

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

pub type ExtraKeys = [u64];

pub fn hash_block_tokens(
    parent_hash: Option<BlockHash>,
    tokens: &[u32],
    extra_keys: &ExtraKeys,
) -> BlockHash {
    let mut hasher = Sha256::new();
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
