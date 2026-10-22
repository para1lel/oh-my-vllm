pub(crate) mod output;
mod qwen38;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelFamily {
    Qwen38,
    Stub,
}
