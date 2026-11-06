use crate::models::qwen38::output::{parameter_value, string_schema};
use anyhow::{Result, ensure};
use serde_json::Value;

#[derive(Clone, Copy)]
enum ToolStage {
    FunctionTag,
    Json {
        depth: usize,
        in_quotes: bool,
        escaped: bool,
    },
    FunctionName {
        start: usize,
    },
    Body,
    ParameterName {
        start: usize,
    },
    ParameterValue {
        start: usize,
        is_string: bool,
        in_quotes: bool,
        escaped: bool,
        matched: usize,
    },
    ToolClose,
}

pub(super) struct ToolProgress {
    cursor: usize,
    stage: ToolStage,
    tool_index: Option<usize>,
    parameter_name: Option<(usize, usize)>,
    #[cfg(test)]
    pub(super) bytes_inspected: usize,
}

impl ToolProgress {
    pub(super) fn new() -> Self {
        Self {
            cursor: "<tool_call>".len(),
            stage: ToolStage::FunctionTag,
            tool_index: None,
            parameter_name: None,
            #[cfg(test)]
            bytes_inspected: 0,
        }
    }

    fn skip_whitespace(&mut self, raw: &str) {
        while let Some(character) = raw[self.cursor..].chars().next() {
            #[cfg(test)]
            {
                self.bytes_inspected += character.len_utf8();
            }
            if !character.is_whitespace() {
                break;
            }
            self.cursor += character.len_utf8();
        }
    }

    fn scan_name(&mut self, raw: &str) -> Option<usize> {
        let tail = &raw[self.cursor..];
        #[cfg(test)]
        {
            self.bytes_inspected += tail.len();
        }
        if let Some(relative) = tail.find('>') {
            let end = self.cursor + relative;
            self.cursor = end + 1;
            Some(end)
        } else {
            self.cursor = raw.len();
            None
        }
    }

    pub(super) fn advance(&mut self, raw: &str, tools: &[Value]) -> Result<Option<usize>> {
        loop {
            let progress = match self.stage {
                ToolStage::FunctionTag => self.function_tag(raw)?,
                ToolStage::FunctionName { start } => self.function_name(raw, start, tools)?,
                ToolStage::Json { .. } => self.json_value(raw)?,
                ToolStage::Body => self.body_tag(raw)?,
                ToolStage::ParameterName { start } => self.parameter_name(raw, start, tools)?,
                ToolStage::ParameterValue { .. } => self.parameter_value(raw, tools)?,
                ToolStage::ToolClose => return self.tool_close(raw),
            };
            if !progress {
                return Ok(None);
            }
        }
    }

    fn function_tag(&mut self, raw: &str) -> Result<bool> {
        self.skip_whitespace(raw);
        let rest = &raw[self.cursor..];
        if rest.starts_with('{') {
            self.cursor += 1;
            self.stage = ToolStage::Json {
                depth: 1,
                in_quotes: false,
                escaped: false,
            };
            return Ok(true);
        }
        #[cfg(test)]
        {
            self.bytes_inspected += rest.len().min("<function=".len());
        }
        if !rest.starts_with("<function=") {
            ensure!("<function=".starts_with(rest), "invalid function tag");
            return Ok(false);
        }
        self.cursor += "<function=".len();
        self.stage = ToolStage::FunctionName { start: self.cursor };
        Ok(true)
    }

    fn function_name(&mut self, raw: &str, start: usize, tools: &[Value]) -> Result<bool> {
        let Some(end) = self.scan_name(raw) else {
            return Ok(false);
        };
        let name = &raw[start..end];
        self.tool_index = Some(
            tools
                .iter()
                .position(|tool| tool["function"]["name"] == name)
                .ok_or_else(|| anyhow::anyhow!("unknown generated function: {name}"))?,
        );
        self.stage = ToolStage::Body;
        Ok(true)
    }

    fn json_value(&mut self, raw: &str) -> Result<bool> {
        let ToolStage::Json {
            mut depth,
            mut in_quotes,
            mut escaped,
        } = self.stage
        else {
            unreachable!()
        };
        while self.cursor < raw.len() && depth > 0 {
            let byte = raw.as_bytes()[self.cursor];
            self.cursor += 1;
            #[cfg(test)]
            {
                self.bytes_inspected += 1;
            }
            if in_quotes {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    in_quotes = false;
                }
            } else {
                match byte {
                    b'"' => in_quotes = true,
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => depth -= 1,
                    _ => {}
                }
            }
        }
        if depth == 0 {
            self.stage = ToolStage::ToolClose;
        } else {
            self.stage = ToolStage::Json {
                depth,
                in_quotes,
                escaped,
            };
            return Ok(false);
        }
        Ok(true)
    }

    fn body_tag(&mut self, raw: &str) -> Result<bool> {
        self.skip_whitespace(raw);
        let rest = &raw[self.cursor..];
        #[cfg(test)]
        {
            self.bytes_inspected += rest.len().min("</function>".len());
            self.bytes_inspected += rest.len().min("<parameter=".len());
        }
        if rest.starts_with("</function>") {
            self.cursor += "</function>".len();
            self.stage = ToolStage::ToolClose;
        } else if rest.starts_with("<parameter=") {
            self.cursor += "<parameter=".len();
            self.stage = ToolStage::ParameterName { start: self.cursor };
        } else {
            ensure!(
                "<parameter=".starts_with(rest) || "</function>".starts_with(rest),
                "invalid parameter tag"
            );
            return Ok(false);
        }
        Ok(true)
    }

    fn parameter_name(&mut self, raw: &str, start: usize, tools: &[Value]) -> Result<bool> {
        let Some(end) = self.scan_name(raw) else {
            return Ok(false);
        };
        let key = &raw[start..end];
        self.parameter_name = Some((start, end));
        let tool = &tools[self.tool_index.expect("function name was resolved")];
        let schema = &tool["function"]["parameters"];
        let is_string = string_schema(&schema["properties"][key], schema, 0)?;
        self.stage = ToolStage::ParameterValue {
            start: self.cursor,
            is_string,
            in_quotes: false,
            escaped: false,
            matched: 0,
        };
        Ok(true)
    }

    fn parameter_value(&mut self, raw: &str, tools: &[Value]) -> Result<bool> {
        const CLOSE_PARAMETER: &[u8] = b"</parameter>";
        let ToolStage::ParameterValue {
            start,
            is_string,
            mut in_quotes,
            mut escaped,
            mut matched,
        } = self.stage
        else {
            unreachable!()
        };
        let mut closed = false;
        while self.cursor < raw.len() {
            let byte = raw.as_bytes()[self.cursor];
            self.cursor += 1;
            #[cfg(test)]
            {
                self.bytes_inspected += 1;
            }
            if !is_string {
                if in_quotes {
                    if escaped {
                        escaped = false;
                    } else if byte == b'\\' {
                        escaped = true;
                    } else if byte == b'"' {
                        in_quotes = false;
                    }
                    continue;
                }
                if byte == b'"' {
                    in_quotes = true;
                    matched = 0;
                    continue;
                }
            }
            matched = if byte == CLOSE_PARAMETER[matched] {
                matched + 1
            } else {
                usize::from(byte == CLOSE_PARAMETER[0])
            };
            if matched == CLOSE_PARAMETER.len() {
                closed = true;
                break;
            }
        }
        if closed {
            if !is_string {
                #[cfg(test)]
                {
                    self.bytes_inspected += self.cursor - start;
                }
                ensure!(
                    self.validate_non_string_value(&raw[..self.cursor], tools)?,
                    "invalid parameter closing tag"
                );
            }
            self.stage = ToolStage::Body;
        } else {
            self.stage = ToolStage::ParameterValue {
                start,
                is_string,
                in_quotes,
                escaped,
                matched,
            };
            return Ok(false);
        }
        Ok(true)
    }

    fn tool_close(&mut self, raw: &str) -> Result<Option<usize>> {
        self.skip_whitespace(raw);
        let rest = &raw[self.cursor..];
        #[cfg(test)]
        {
            self.bytes_inspected += rest.len().min("</tool_call>".len());
        }
        if !rest.starts_with("</tool_call>") {
            ensure!("</tool_call>".starts_with(rest), "invalid tool closing tag");
            return Ok(None);
        }
        self.cursor += "</tool_call>".len();
        Ok(Some(self.cursor))
    }

    pub(super) fn validate_non_string_value(&self, raw: &str, tools: &[Value]) -> Result<bool> {
        let ToolStage::ParameterValue {
            start,
            is_string: false,
            ..
        } = self.stage
        else {
            return Ok(false);
        };
        let (name_start, name_end) = self.parameter_name.expect("parameter name was scanned");
        let schema =
            &tools[self.tool_index.expect("function name was resolved")]["function"]["parameters"];
        let parsed = parameter_value(
            &raw[start..],
            &schema["properties"][&raw[name_start..name_end]],
            schema,
        )?;
        Ok(parsed.is_some())
    }
}
