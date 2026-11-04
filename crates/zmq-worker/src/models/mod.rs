mod cache;
pub(crate) mod deepseek_v41;
pub use cache::ServiceCache;
pub use deepseek_v41::cache::Dep4Cache;
pub(crate) mod execution;
pub(crate) mod output;
mod policy;
mod qwen38;
mod stub;

pub use policy::{ModelDefaults, ModelPolicy, SpeculativeMode, select_model};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelFamily {
    DeepSeekV41,
    Qwen38,
    Stub,
}
