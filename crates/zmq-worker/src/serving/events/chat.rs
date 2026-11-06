use crate::models::output::Delta;
use crate::serving::events::Output;
use serde_json::{Value, json};

impl Output {
    pub(super) fn chat_delta(&mut self, delta: Delta) -> Vec<Value> {
        let wire = match delta {
            Delta::Text(text) => {
                self.content.push_str(&text);
                json!({
                    "content": text,
                })
            }
            Delta::Reasoning(text) => {
                self.reasoning_text.push_str(&text);
                json!({
                    "reasoning_content": text,
                })
            }
            Delta::Tool {
                index,
                name,
                arguments,
            } => {
                let call = json!({
                    "id": format!("call_{}_{}",self.id,index),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                });
                self.calls.push(call.clone());
                let mut c = call;
                c["index"] = json!(index);
                json!({
                    "tool_calls": [
                        c,
                    ],
                })
            }
        };
        vec![self.chat_chunk(wire, Value::Null)]
    }

    pub(super) fn finish_chat(&mut self, reason: &str) -> (Value, Vec<Value>) {
        let finish = if reason == "length" {
            "length"
        } else if !self.calls.is_empty() {
            "tool_calls"
        } else {
            "stop"
        };
        let mut events = vec![self.chat_chunk(json!({}), json!(finish))];
        if self.request.include_usage {
            events.push(json!({
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.request.model,
                "choices": [],
                "usage": self.usage(),
            }));
        }
        let mut message = json!({
            "role": "assistant",
            "content": if self.content.is_empty() {Value::Null} else {json!(self.content)},
        });
        if !self.reasoning_text.is_empty() {
            message["reasoning_content"] = json!(self.reasoning_text);
        }
        if !self.calls.is_empty() {
            message["tool_calls"] = json!(self.calls);
        }
        let result = json!({
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.request.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish,
                },
            ],
            "usage": self.usage(),
        });
        (result, events)
    }
}
