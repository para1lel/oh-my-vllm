use anyhow::Result;
use serde_json::Value;

use super::{ModelFamily, deepseek_v41, qwen38};

#[derive(Debug, Clone)]
pub enum Delta {
    Reasoning(String),
    Text(String),
    Tool {
        index: usize,
        name: String,
        arguments: String,
    },
}

pub(crate) enum Parser {
    DeepSeekV41(deepseek_v41::output::Parser),
    Qwen38(qwen38::output::Parser),
}

impl Parser {
    pub(crate) fn new(family: ModelFamily, thinking: bool, tools: Vec<Value>) -> Self {
        match family {
            ModelFamily::DeepSeekV41 => Self::DeepSeekV41(deepseek_v41::output::Parser::new(
                thinking,
                !tools.is_empty(),
            )),
            ModelFamily::Qwen38 => Self::Qwen38(qwen38::output::Parser::new(thinking, tools)),
            // The explicit test backend simulates the existing Qwen response protocol.
            ModelFamily::Stub => Self::Qwen38(qwen38::output::Parser::new(thinking, tools)),
        }
    }

    pub(crate) fn feed(&mut self, text: &str, finished: bool) -> Result<Vec<Delta>> {
        match self {
            Self::DeepSeekV41(parser) => parser.feed(text, finished),
            Self::Qwen38(parser) => parser.feed(text, finished),
        }
    }

    pub(crate) fn feed_length(&mut self, text: &str) -> Result<Vec<Delta>> {
        match self {
            Self::DeepSeekV41(parser) => parser.feed_length(text),
            Self::Qwen38(parser) => parser.feed_length(text),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deepseek_parser_keeps_text_capability_without_tool_grammar() {
        let mut parser = Parser::new(ModelFamily::DeepSeekV41, false, Vec::new());
        assert!(matches!(
            parser.feed("hello", false).unwrap().as_slice(),
            [Delta::Text(text)] if text == "hello"
        ));
        let mut parser = Parser::new(ModelFamily::DeepSeekV41, false, vec![json!({})]);
        assert!(parser.feed("hello", false).is_err());
    }
}
