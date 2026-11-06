use anyhow::{Result, bail, ensure};
use serde::Deserialize;
use serde_json::Value;

use crate::models::output::Delta;

mod tool_progress;

use tool_progress::ToolProgress;

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
        let body = raw.strip_prefix("<tool_call>").unwrap_or(raw).trim_start();
        if body.starts_with('{') {
            let body = body
                .strip_suffix("</tool_call>")
                .ok_or_else(|| anyhow::anyhow!("missing tool close"))?;
            let call: Value = serde_json::from_str(body)?;
            let name = call["name"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing tool name"))?;
            ensure!(
                self.tools
                    .iter()
                    .any(|tool| tool["function"]["name"] == name),
                "unknown generated function: {name}"
            );
            ensure!(
                call["arguments"].is_object(),
                "tool arguments must be an object"
            );
            return Ok((name.to_owned(), call["arguments"].to_string()));
        }
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
    use serde_json::json;

    #[test]
    fn json_tool_frames_preserve_strings_across_stream_fragments() {
        let mut parser = Parser::new(false, vec![json!({"function":{"name":"echo"}})]);
        let arguments = json!({"text":"中\n</tool_call>\\\"", "nested":[{"ok":true}]});
        let raw = format!(
            "<tool_call>{}</tool_call>",
            json!({"name":"echo", "arguments":arguments})
        );
        let mut events = Vec::new();
        for character in raw.chars() {
            events.extend(parser.feed(&character.to_string(), false).unwrap());
        }
        assert_eq!(events.len(), 1);
        let Delta::Tool {
            name,
            arguments: actual,
            ..
        } = &events[0]
        else {
            panic!("expected a tool call");
        };
        assert_eq!(name, "echo");
        assert_eq!(serde_json::from_str::<Value>(actual).unwrap(), arguments);
        assert!(parser.feed("", true).unwrap().is_empty());
    }
}
