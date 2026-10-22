use anyhow::Result;
use serde_json::Value;

use super::{ModelFamily, qwen38};

#[derive(Debug, Clone)]
pub(crate) enum Delta {
    Reasoning(String),
    Text(String),
    Tool {
        index: usize,
        name: String,
        arguments: String,
    },
}

pub(crate) enum Parser {
    Qwen38(qwen38::output::Parser),
}

impl Parser {
    pub(crate) fn new(family: ModelFamily, thinking: bool, tools: Vec<Value>) -> Self {
        match family {
            ModelFamily::Qwen38 => Self::Qwen38(qwen38::output::Parser::new(thinking, tools)),
            // The explicit test backend simulates the existing Qwen response protocol.
            ModelFamily::Stub => Self::Qwen38(qwen38::output::Parser::new(thinking, tools)),
        }
    }

    pub(crate) fn feed(&mut self, text: &str, finished: bool) -> Result<Vec<Delta>> {
        match self {
            Self::Qwen38(parser) => parser.feed(text, finished),
        }
    }

    pub(crate) fn feed_length(&mut self, text: &str) -> Result<Vec<Delta>> {
        match self {
            Self::Qwen38(parser) => parser.feed_length(text),
        }
    }
}
