pub(crate) mod output;
mod policy;
mod qwen38;
mod stub;

pub use policy::{ModelDefaults, ModelPolicy, SpeculativeMode, select_model};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelFamily {
    Qwen38,
    Stub,
}
