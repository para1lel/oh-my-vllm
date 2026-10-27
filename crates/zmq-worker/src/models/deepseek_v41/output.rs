use anyhow::{Result, bail};

use super::super::output::Delta;

pub(crate) struct Parser {
    thinking: bool,
    pending: String,
    has_tools: bool,
}

impl Parser {
    pub(crate) fn new(thinking: bool, has_tools: bool) -> Self {
        Self {
            thinking,
            pending: String::new(),
            has_tools,
        }
    }

    pub(crate) fn feed(&mut self, text: &str, finished: bool) -> Result<Vec<Delta>> {
        if self.has_tools {
            bail!("DeepSeek tools are not enabled by the selected capability set");
        }
        self.pending.push_str(text);
        if self.thinking {
            if let Some(end) = self.pending.find("</think>") {
                let reasoning = self.pending[..end].to_owned();
                self.pending.drain(..end + "</think>".len());
                self.thinking = false;
                let mut events = Vec::new();
                if !reasoning.is_empty() {
                    events.push(Delta::Reasoning(reasoning));
                }
                if finished && !self.pending.is_empty() {
                    events.push(Delta::Text(std::mem::take(&mut self.pending)));
                }
                return Ok(events);
            }
            if finished {
                self.thinking = false;
                return Ok(vec![Delta::Reasoning(std::mem::take(&mut self.pending))]);
            }
            return Ok(Vec::new());
        }
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }
        Ok(vec![Delta::Text(std::mem::take(&mut self.pending))])
    }

    pub(crate) fn feed_length(&mut self, text: &str) -> Result<Vec<Delta>> {
        self.feed(text, true)
    }
}
