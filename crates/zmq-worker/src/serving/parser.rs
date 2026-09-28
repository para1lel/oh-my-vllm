//! Incremental Qwen text/reasoning/XML parsing, independent of HTTP framing.
use anyhow::{Result, bail, ensure};
use serde::Deserialize;
use serde_json::Value;
#[cfg(test)]
use serde_json::json;

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

#[derive(Clone, Copy)]
enum ToolStage {
    FunctionTag,
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

/// Byte offsets always point to an ASCII tag boundary or the end of a UTF-8 chunk.
struct ToolProgress {
    cursor: usize,
    stage: ToolStage,
    tool_index: Option<usize>,
    parameter_name: Option<(usize, usize)>,
    #[cfg(test)]
    bytes_inspected: usize,
}

impl ToolProgress {
    fn new() -> Self {
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

    fn advance(&mut self, raw: &str, tools: &[Value]) -> Result<Option<usize>> {
        const CLOSE_PARAMETER: &[u8] = b"</parameter>";
        loop {
            match self.stage {
                ToolStage::FunctionTag => {
                    self.skip_whitespace(raw);
                    let rest = &raw[self.cursor..];
                    #[cfg(test)]
                    {
                        self.bytes_inspected += rest.len().min("<function=".len());
                    }
                    if !rest.starts_with("<function=") {
                        ensure!("<function=".starts_with(rest), "invalid function tag");
                        return Ok(None);
                    }
                    self.cursor += "<function=".len();
                    self.stage = ToolStage::FunctionName { start: self.cursor };
                }
                ToolStage::FunctionName { start } => {
                    let Some(end) = self.scan_name(raw) else {
                        return Ok(None);
                    };
                    let name = &raw[start..end];
                    self.tool_index = Some(
                        tools
                            .iter()
                            .position(|tool| tool["function"]["name"] == name)
                            .ok_or_else(|| anyhow::anyhow!("unknown generated function: {name}"))?,
                    );
                    self.stage = ToolStage::Body;
                }
                ToolStage::Body => {
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
                        return Ok(None);
                    }
                }
                ToolStage::ParameterName { start } => {
                    let Some(end) = self.scan_name(raw) else {
                        return Ok(None);
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
                }
                ToolStage::ParameterValue {
                    start,
                    is_string,
                    mut in_quotes,
                    mut escaped,
                    mut matched,
                } => {
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
                        return Ok(None);
                    }
                }
                ToolStage::ToolClose => {
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
                    return Ok(Some(self.cursor));
                }
            }
        }
    }

    fn validate_non_string_value(&self, raw: &str, tools: &[Value]) -> Result<bool> {
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

pub struct Parser {
    pending: String,
    thinking: bool,
    tools: Vec<Value>,
    tool_progress: Option<ToolProgress>,
    #[cfg(test)]
    completed_tool_scan_bytes: usize,
    pub calls: usize,
}

impl Parser {
    pub fn new(thinking: bool, tools: Vec<Value>) -> Self {
        Self {
            pending: String::new(),
            thinking,
            tools,
            tool_progress: None,
            #[cfg(test)]
            completed_tool_scan_bytes: 0,
            calls: 0,
        }
    }

    pub fn feed(&mut self, text: &str, finished: bool) -> Result<Vec<Delta>> {
        self.feed_inner(text, finished, false)
    }

    pub fn feed_length(&mut self, text: &str) -> Result<Vec<Delta>> {
        self.feed_inner(text, true, true)
    }

    fn feed_inner(
        &mut self,
        text: &str,
        finished: bool,
        allow_incomplete_tool: bool,
    ) -> Result<Vec<Delta>> {
        self.pending.push_str(text);
        let mut events = Vec::new();
        loop {
            if self.thinking {
                if let Some(end) = self.pending.find("</think>") {
                    events.push(Delta::Reasoning(self.pending[..end].to_owned()));
                    self.pending.drain(..end + 8);
                    self.thinking = false;
                    continue;
                }
                let count = if finished {
                    self.pending.len()
                } else {
                    safe_prefix(&self.pending, "</think>")
                };
                if count > 0 {
                    events.push(Delta::Reasoning(self.pending.drain(..count).collect()));
                }
                break;
            }
            if self.tools.is_empty() {
                if !self.pending.is_empty() {
                    events.push(Delta::Text(std::mem::take(&mut self.pending)));
                }
                break;
            }
            if self.pending.starts_with("<tool_call>") {
                let progress = self.tool_progress.get_or_insert_with(ToolProgress::new);
                let Some(end) = progress.advance(&self.pending, &self.tools)? else {
                    if finished {
                        if allow_incomplete_tool {
                            progress.validate_non_string_value(&self.pending, &self.tools)?;
                            self.pending.clear();
                            self.tool_progress = None;
                            break;
                        }
                        bail!("incomplete tool call at generation end");
                    }
                    break;
                };
                #[cfg(test)]
                {
                    self.completed_tool_scan_bytes += progress.bytes_inspected;
                }
                self.tool_progress = None;
                let raw: String = self.pending.drain(..end).collect();
                let (name, arguments) = self.parse_tool(&raw)?;
                events.push(Delta::Tool {
                    index: self.calls,
                    name,
                    arguments,
                });
                self.calls += 1;
                continue;
            }
            let count = if let Some(start) = self.pending.find("<tool_call>") {
                start
            } else if finished {
                self.pending.len()
            } else {
                safe_prefix(&self.pending, "<tool_call>")
            };
            if count == 0 {
                break;
            }
            events.push(Delta::Text(self.pending.drain(..count).collect()));
        }
        Ok(events)
    }

    fn parse_tool(&self, raw: &str) -> Result<(String, String)> {
        let function = raw
            .split_once("<function=")
            .ok_or_else(|| anyhow::anyhow!("missing function tag"))?
            .1;
        let (name, body) = function
            .split_once('>')
            .ok_or_else(|| anyhow::anyhow!("missing function name"))?;
        let tool = self
            .tools
            .iter()
            .find(|t| t["function"]["name"] == name)
            .ok_or_else(|| anyhow::anyhow!("unknown generated function: {name}"))?;
        let schema = &tool["function"]["parameters"];
        let properties = &schema["properties"];
        let mut rest = body.trim();
        let mut args = serde_json::Map::new();
        while !rest.starts_with("</function>") {
            ensure!(
                rest.starts_with("<parameter="),
                "invalid tool parameter syntax"
            );
            let (key, value) = rest[11..]
                .split_once('>')
                .ok_or_else(|| anyhow::anyhow!("missing parameter name"))?;
            let property = resolve_schema(&properties[key], schema, 0)?;
            let (value, tail) = parameter_value(value, property, schema)?
                .ok_or_else(|| anyhow::anyhow!("missing parameter close"))?;
            ensure!(!args.contains_key(key), "duplicate tool parameter");
            let parsed = if string_schema(property, schema, 0)? {
                Value::String(restore_string(value, property, schema)?)
            } else {
                serde_json::from_str(value)?
            };
            args.insert(key.to_owned(), parsed);
            rest = tail.trim();
        }
        Ok((name.to_owned(), Value::Object(args).to_string()))
    }
}

// JSON containers may contain XML delimiters inside strings. Deserialize exactly
// one JSON value before examining the XML suffix; raw strings use the delimiter.
fn parameter_value<'a>(
    value: &'a str,
    property: &Value,
    root: &Value,
) -> Result<Option<(&'a str, &'a str)>> {
    const CLOSE: &str = "</parameter>";
    if string_schema(property, root, 0)? {
        return Ok(value.split_once(CLOSE));
    }
    let mut deserializer = serde_json::Deserializer::from_str(value);
    match Value::deserialize(&mut deserializer) {
        Ok(_) => {}
        Err(error) if error.is_eof() => return Ok(None),
        Err(error) => return Err(error.into()),
    }
    // into_iter starts its byte offset at the already consumed value. Calling
    // next here would reject '<' as a JSON-stream separator after scalar values.
    let end = deserializer.into_iter::<Value>().byte_offset();
    let suffix = value[end..].trim_start_matches([' ', '\n', '\t', '\r']);
    if let Some(tail) = suffix.strip_prefix(CLOSE) {
        return Ok(Some((&value[..end], tail)));
    }
    ensure!(CLOSE.starts_with(suffix), "invalid parameter closing tag");
    Ok(None)
}

fn string_candidates(schema: &Value, root: &Value, depth: usize) -> Result<Option<Vec<String>>> {
    ensure!(depth < 32, "recursive tool parameter type");
    let schema = resolve_schema(schema, root, depth)?;
    if let Some(value) = schema["const"].as_str() {
        return Ok(Some(vec![value.to_owned()]));
    }
    if let Some(values) = schema["enum"].as_array() {
        return Ok(Some(
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect(),
        ));
    }
    if let Some(alternatives) = schema["anyOf"].as_array() {
        let mut values = Vec::new();
        for alternative in alternatives {
            let Some(candidates) = string_candidates(alternative, root, depth + 1)? else {
                return Ok(None);
            };
            values.extend(candidates);
        }
        return Ok(Some(values));
    }
    Ok(None)
}

fn restore_string(value: &str, schema: &Value, root: &Value) -> Result<String> {
    let Some(candidates) = string_candidates(schema, root, 0)? else {
        // Qwen's XML template wraps parameter values in one newline on each
        // side (vLLM parser/qwen3.py::_trim_wrapping_newlines). Preserve spaces,
        // quotes and any additional newlines belonging to the actual value.
        let value = value.strip_prefix('\n').unwrap_or(value);
        return Ok(value.strip_suffix('\n').unwrap_or(value).to_owned());
    };
    let mut matches: Vec<_> = candidates
        .into_iter()
        .filter(|candidate| {
            value.match_indices(candidate.as_str()).any(|(start, _)| {
                value[..start]
                    .chars()
                    .all(|c| matches!(c, ' ' | '\n' | '\t'))
                    && value[start + candidate.len()..]
                        .chars()
                        .all(|c| matches!(c, ' ' | '\n' | '\t'))
            })
        })
        .collect();
    matches.sort();
    matches.dedup();
    ensure!(
        matches.len() == 1,
        "ambiguous or invalid XML string enum/const padding"
    );
    Ok(matches.pop().unwrap())
}
fn resolve_schema<'a>(schema: &'a Value, root: &'a Value, depth: usize) -> Result<&'a Value> {
    ensure!(depth < 32, "recursive tool parameter type");
    if let Some(reference) = schema["$ref"].as_str() {
        let path = reference
            .strip_prefix('#')
            .ok_or_else(|| anyhow::anyhow!("external schema reference"))?;
        return resolve_schema(
            root.pointer(path)
                .ok_or_else(|| anyhow::anyhow!("missing schema reference"))?,
            root,
            depth + 1,
        );
    }
    Ok(schema)
}
fn string_schema(schema: &Value, root: &Value, depth: usize) -> Result<bool> {
    ensure!(depth < 32, "recursive tool parameter type");
    let schema = resolve_schema(schema, root, depth)?;
    if schema["type"] == "string"
        || schema["type"]
            .as_array()
            .is_some_and(|a| !a.is_empty() && a.iter().all(|t| t == "string"))
    {
        return Ok(true);
    }
    if let Some(alternatives) = schema["anyOf"].as_array() {
        return alternatives
            .iter()
            .try_fold(true, |all, s| Ok(all && string_schema(s, root, depth + 1)?));
    }
    Ok(false)
}

fn safe_prefix(text: &str, marker: &str) -> usize {
    let hold = (1..marker.len())
        .filter(|&n| text.ends_with(&marker[..n]))
        .max()
        .unwrap_or(0);
    text.len() - hold
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fragmented_reasoning_and_typed_tool() {
        let mut parser = Parser::new(
            true,
            vec![json!({
                "function": {
                    "name": "read",
                    "parameters": {
                        "properties": {
                            "path": {
                                "type": "string",
                            },
                            "n": {
                                "type": "integer",
                            },
                        },
                    },
                },
            })],
        );
        let raw = concat!(
            "想一想</think>\n<tool_call>\n<function=read>\n",
            "<parameter=path>README.md</parameter>\n",
            "<parameter=n>2</parameter>\n</function>\n</tool_call>"
        );
        let mut reasoning = String::new();
        let mut calls = Vec::new();
        for c in raw.chars() {
            for event in parser.feed(&c.to_string(), false).unwrap() {
                match event {
                    Delta::Reasoning(t) => reasoning.push_str(&t),
                    Delta::Tool {
                        name, arguments, ..
                    } => calls.push((name, arguments)),
                    _ => {}
                }
            }
        }
        parser.feed("", true).unwrap();
        assert_eq!(reasoning, "想一想");
        assert_eq!(calls[0].0, "read");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].1).unwrap(),
            json!({
                "path": "README.md",
                "n": 2,
            })
        );
    }
    #[test]
    fn literals_are_not_tool_calls_without_tools() {
        let mut parser = Parser::new(false, vec![]);
        let raw = r#"{"text":"<tool_call>"}"#;
        let events = parser.feed(raw, true).unwrap();
        assert!(matches!(&events[0], Delta::Text(text) if text == raw));
    }
    #[test]
    fn tool_strings_preserve_quotes_newlines_and_close_tag_literals() {
        let mut parser = Parser::new(
            false,
            vec![json!({
                "function": {
                    "name": "read",
                    "parameters": {
                        "properties": {
                            "path": {
                                "type": "string",
                            },
                        },
                    },
                },
            })],
        );
        let raw = concat!(
            "<tool_call><function=read><parameter=path>\n\n",
            "\"hello\" </function> </tool_call>\n\n",
            "</parameter></function></tool_call>"
        );
        let events = parser.feed(raw, true).unwrap();
        let Delta::Tool { arguments, .. } = &events[0] else {
            panic!("expected tool")
        };
        assert_eq!(
            serde_json::from_str::<Value>(arguments).unwrap()["path"],
            "\n\"hello\" </function> </tool_call>\n"
        );
    }
    #[test]
    fn incomplete_tool_is_an_error() {
        let mut parser = Parser::new(
            false,
            vec![json!({
                "function": {
                    "name": "read",
                },
            })],
        );
        assert!(parser.feed("<tool_call><function=read>", true).is_err());
    }

    #[test]
    fn length_finish_discards_incomplete_tool() {
        let mut parser = Parser::new(false, vec![json!({"function": {"name": "read"}})]);
        let deltas = parser
            .feed_length("prefix<tool_call><function=read>")
            .unwrap();
        assert_eq!(deltas.len(), 1);
        assert!(matches!(&deltas[0], Delta::Text(text) if text == "prefix"));
        assert!(parser.pending.is_empty());
    }

    #[test]
    fn fragmented_nested_json_preserves_xml_delimiters() {
        let mut parser = Parser::new(
            false,
            vec![json!({
                "function": {
                    "name": "test",
                    "parameters": {
                        "properties": {
                            "object": {
                                "type": "object",
                            },
                            "array": {
                                "type": "array",
                            },
                            "number": {
                                "type": "number",
                            },
                            "flag": {
                                "type": "boolean",
                            },
                        },
                    },
                },
            })],
        );
        let raw = concat!(
            "<tool_call><function=test>",
            "<parameter=object>{\"nested\":{\"text\":\"",
            "</parameter></function></tool_call>\"}}</parameter>",
            "<parameter=array>[\"</parameter>\", {\"x\":\"\\\"quoted\\\"\"}]</parameter>",
            "<parameter=number>-12.5e2</parameter>",
            "<parameter=flag>true</parameter></function></tool_call>"
        );
        let mut events = Vec::new();
        for character in raw.chars() {
            events.extend(parser.feed(&character.to_string(), false).unwrap());
        }
        events.extend(parser.feed("", true).unwrap());
        let Delta::Tool { arguments, .. } = &events[0] else {
            panic!("expected tool")
        };
        assert_eq!(
            serde_json::from_str::<Value>(arguments).unwrap(),
            json!({
                "object": {
                    "nested": {
                        "text": "</parameter></function></tool_call>",
                    },
                },
                "array": [
                    "</parameter>",
                    {
                        "x": "\"quoted\"",
                    },
                ],
                "number": -1250.0,
                "flag": true,
            })
        );
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn tool_scan_is_linear_for_all_fragment_sizes() {
        let tools = vec![json!({
            "function": {
                "name": "test",
                "parameters": {
                    "properties": {
                        "object": {"type": "object"},
                        "tail": {"type": "string"},
                    },
                },
            },
        })];
        let object = json!({
            "text": format!("{} </parameter> \\ \"", "é中".repeat(2048)),
        });
        let raw = format!(
            "<tool_call>\n<function=test><parameter=object>{object}</parameter>\n\
             <parameter=tail>done</parameter></function>\n</tool_call>"
        );
        let characters: Vec<char> = raw.chars().collect();
        for chunk_size in [1, 2, 3, 7, 16, 127, usize::MAX] {
            let mut parser = Parser::new(false, tools.clone());
            let mut events = Vec::new();
            for chunk in characters.chunks(chunk_size) {
                let fragment: String = chunk.iter().collect();
                events.extend(parser.feed(&fragment, false).unwrap());
            }
            assert!(
                parser.completed_tool_scan_bytes <= raw.len() * 6,
                "chunk_size={chunk_size}, bytes_inspected={}, input_bytes={}",
                parser.completed_tool_scan_bytes,
                raw.len()
            );
            assert!(parser.tool_progress.is_none());
            assert_eq!(events.len(), 1, "chunk_size={chunk_size}");
            let Delta::Tool {
                index,
                name,
                arguments,
            } = &events[0]
            else {
                panic!("expected a tool call")
            };
            assert_eq!((*index, name.as_str()), (0, "test"));
            assert_eq!(
                serde_json::from_str::<Value>(arguments).unwrap(),
                json!({"object": object, "tail": "done"})
            );
        }
    }

    #[test]
    fn fragmented_incomplete_tool_markers_preserve_finish_contract() {
        let tools = vec![json!({"function": {"name": "read"}})];
        for marker in ["<tool_", "<tool_call><funct", "<tool_call><function=rea"] {
            let mut parser = Parser::new(false, tools.clone());
            for character in marker.chars() {
                parser.feed(&character.to_string(), false).unwrap();
            }
            let mut length_parser = Parser::new(false, tools.clone());
            for character in marker.chars() {
                length_parser.feed(&character.to_string(), false).unwrap();
            }
            if marker == "<tool_" {
                assert!(matches!(
                    &parser.feed("", true).unwrap()[..],
                    [Delta::Text(text)] if text == marker
                ));
            } else {
                assert!(parser.feed("", true).is_err(), "marker={marker}");
            }
            length_parser.feed_length("").unwrap();
            assert!(length_parser.pending.is_empty());
            assert!(length_parser.tool_progress.is_none());
        }
    }

    #[test]
    fn length_finish_rejects_invalid_partial_json_parameter() {
        let tools = vec![json!({
            "function": {
                "name": "read",
                "parameters": {"properties": {"count": {"type": "integer"}}},
            },
        })];
        for (value, invalid) in [
            ("truX", true),
            ("1x", true),
            ("{\"x\": }", true),
            ("tru", false),
            ("1</para", false),
        ] {
            let mut parser = Parser::new(false, tools.clone());
            let raw = format!("<tool_call><function=read><parameter=count>{value}");
            for character in raw.chars() {
                parser.feed(&character.to_string(), false).unwrap();
            }
            assert_eq!(parser.feed_length("").is_err(), invalid, "value={value}");
        }
    }

    #[test]
    fn closed_parameter_is_validated_before_incomplete_tool_is_discarded() {
        let tools = vec![json!({
            "function": {
                "name": "read",
                "parameters": {"properties": {"count": {"type": "integer"}}},
            },
        })];
        for value in ["truX", "1x", "{\"x\": }"] {
            let raw = format!(
                "<tool_call><function=read><parameter=count>{value}</parameter></function>"
            );
            for chunk_size in [1, 3, usize::MAX] {
                let mut parser = Parser::new(false, tools.clone());
                let characters: Vec<char> = raw.chars().collect();
                let mut error = false;
                for chunk in characters.chunks(chunk_size) {
                    let fragment: String = chunk.iter().collect();
                    if parser.feed(&fragment, false).is_err() {
                        error = true;
                        break;
                    }
                }
                if !error {
                    error = parser.feed_length("").is_err();
                }
                assert!(error, "value={value}, chunk_size={chunk_size}");
            }
        }
    }

    #[test]
    fn enum_const_padding_restores_schema_values() {
        let root = json!({
            "$defs": {
                "word": {
                    "type": "string",
                    "const": "\nabc\n",
                },
            },
        });
        assert_eq!(
            restore_string(
                " \nabc\n\t",
                &json!({
                    "$ref": "#/$defs/word",
                }),
                &root
            )
            .unwrap(),
            "\nabc\n"
        );
        assert_eq!(
            restore_string(
                " \tabc\n",
                &json!({
                    "type": "string",
                    "enum": [
                        "abc",
                        "def",
                    ],
                }),
                &root
            )
            .unwrap(),
            "abc"
        );
        assert_eq!(
            restore_string(
                "\tdef ",
                &json!({
                    "anyOf": [
                        {
                            "type": "string",
                            "const": "abc",
                        },
                        {
                            "type": "string",
                            "const": "def",
                        },
                    ],
                }),
                &root
            )
            .unwrap(),
            "def"
        );
        assert!(
            restore_string(
                " abc ",
                &json!({
                    "type": "string",
                    "enum": [
                        "abc",
                        " abc",
                    ],
                }),
                &root
            )
            .is_err()
        );
        assert!(
            restore_string(
                "abc",
                &json!({
                    "type": "string",
                    "const": "\nabc\n",
                }),
                &root
            )
            .is_err()
        );
        assert_eq!(
            restore_string(
                " \n",
                &json!({
                    "type": "string",
                    "const": "",
                }),
                &root
            )
            .unwrap(),
            ""
        );
    }

    #[test]
    fn length_finish_flushes_pending_marker_prefix() {
        // SRV-04 regression: reasoning ends in a prefix of </think> while
        // max_tokens is exhausted (reason.is_some() but reason != "stop").
        // The pending bytes must not be silently dropped.
        let mut parser = Parser::new(true, vec![]);
        let events = parser.feed("hello</th", true).unwrap();
        let mut reasoning = String::new();
        for e in events {
            if let Delta::Reasoning(t) = e {
                reasoning.push_str(&t);
            }
        }
        assert_eq!(reasoning, "hello</th");
    }

    #[test]
    fn length_finish_flushes_pending_tool_marker_prefix() {
        let mut parser = Parser::new(
            false,
            vec![json!({
                "function": {"name": "read", "parameters": {"type": "object"}}
            })],
        );
        let first = parser.feed("hello<tool_cal", false).unwrap();
        assert!(matches!(&first[..], [Delta::Text(text)] if text == "hello"));
        let last = parser.feed_length("").unwrap();
        assert!(matches!(&last[..], [Delta::Text(text)] if text == "<tool_cal"));
        assert!(parser.pending.is_empty());
    }
}
