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

pub struct Parser {
    pending: String,
    thinking: bool,
    tools: Vec<Value>,
    pub calls: usize,
}

impl Parser {
    pub fn new(thinking: bool, tools: Vec<Value>) -> Self {
        Self {
            pending: String::new(),
            thinking,
            tools,
            calls: 0,
        }
    }

    pub fn feed(&mut self, text: &str, finished: bool) -> Result<Vec<Delta>> {
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
                let Some(end) = tool_end(&self.pending, &self.tools)? else {
                    if finished {
                        bail!("incomplete tool call at generation end");
                    }
                    break;
                };
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

// Close tags inside a raw parameter are ordinary string content. Only recognize
// function/tool boundaries after the corresponding parameter has ended.
fn tool_end(raw: &str, tools: &[Value]) -> Result<Option<usize>> {
    let mut rest = raw.strip_prefix("<tool_call>").unwrap_or(raw).trim_start();
    if "<function=".starts_with(rest) {
        return Ok(None);
    }
    ensure!(rest.starts_with("<function="), "invalid function tag");
    let Some((name, tail)) = rest[10..].split_once('>') else {
        return Ok(None);
    };
    let tool = tools
        .iter()
        .find(|tool| tool["function"]["name"] == name)
        .ok_or_else(|| anyhow::anyhow!("unknown generated function: {name}"))?;
    let schema = &tool["function"]["parameters"];
    rest = tail.trim_start();
    loop {
        if rest.starts_with("</function>") {
            rest = rest[11..].trim_start();
            if rest.starts_with("</tool_call>") {
                return Ok(Some(raw.len() - rest.len() + 12));
            }
            ensure!("</tool_call>".starts_with(rest), "invalid tool closing tag");
            return Ok(None);
        }
        if "<parameter=".starts_with(rest) || "</function>".starts_with(rest) {
            return Ok(None);
        }
        ensure!(rest.starts_with("<parameter="), "invalid parameter tag");
        let Some((key, value)) = rest[11..].split_once('>') else {
            return Ok(None);
        };
        let Some((_, tail)) = parameter_value(value, &schema["properties"][key], schema)? else {
            return Ok(None);
        };
        rest = tail.trim_start();
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
}
