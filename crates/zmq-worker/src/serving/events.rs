mod chat;
mod completion;
mod delta;

use super::request::Request;
use crate::models::output::Delta;
use serde_json::{Value, json};

pub struct Output {
    pub id: String,
    pub request: Request,
    created: u64,
    output: Vec<Value>,
    reasoning: Option<usize>,
    message: Option<usize>,
    sequence: u64,
    pub content: String,
    pub reasoning_text: String,
    pub calls: Vec<Value>,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub reasoning_tokens: usize,
    pub cached_tokens: usize,
}

impl Output {
    pub fn new(id: String, request: Request) -> Self {
        Self {
            id,
            request,
            created: super::now(),
            output: vec![],
            reasoning: None,
            message: None,
            sequence: 0,
            content: String::new(),
            reasoning_text: String::new(),
            calls: vec![],
            input_tokens: 0,
            output_tokens: 0,
            reasoning_tokens: 0,
            cached_tokens: 0,
        }
    }

    fn event(&mut self, kind: &str, mut body: Value) -> Value {
        body["type"] = json!(kind);
        body["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        body
    }

    fn chat_chunk(&self, delta: Value, finish: Value) -> Value {
        json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish,
                },
            ],
        })
    }

    pub fn start(&mut self) -> Vec<Value> {
        if !self.request.responses {
            return vec![self.chat_chunk(
                json!({
                    "role": "assistant",
                    "content": "",
                }),
                Value::Null,
            )];
        }
        let response = self.response("in_progress", None);
        vec![
            self.event(
                "response.created",
                json!({
                    "response": response,
                }),
            ),
            self.event(
                "response.in_progress",
                json!({
                    "response": response,
                }),
            ),
        ]
    }

    pub fn delta(&mut self, delta: Delta) -> Vec<Value> {
        if !self.request.responses {
            return self.chat_delta(delta);
        }
        let mut events = Vec::new();
        match delta {
            Delta::Reasoning(text) => self.reasoning_delta(text, &mut events),
            Delta::Text(text) => self.text_delta(text, &mut events),
            Delta::Tool {
                index,
                name,
                arguments,
            } => self.tool_delta(index, name, arguments, &mut events),
        }
        events
    }

    pub fn finish(&mut self, reason: &str) -> (Value, Vec<Value>) {
        if !self.request.responses {
            return self.finish_chat(reason);
        }
        let mut events = Vec::new();
        for i in 0..self.output.len() {
            let item = self.output[i].clone();
            self.finish_item(i, &item, reason, &mut events);
            events.push(self.event(
                "response.output_item.done",
                json!({
                    "output_index": i,
                    "item": self.output[i],
                }),
            ));
        }
        let status = if reason == "length" {
            "incomplete"
        } else {
            "completed"
        };
        let response = self.response(status, Some(reason));
        events.push(self.event(
            if reason == "length" {
                "response.incomplete"
            } else {
                "response.completed"
            },
            json!({
                "response": response,
            }),
        ));
        (response, events)
    }

    fn usage(&self) -> Value {
        if self.request.responses {
            json!({
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens+self.output_tokens,
                "input_tokens_details": {
                    "cached_tokens": self.cached_tokens,
                },
                "output_tokens_details": {
                    "reasoning_tokens": self.reasoning_tokens,
                },
            })
        } else {
            json!({
                "prompt_tokens": self.input_tokens,
                "completion_tokens": self.output_tokens,
                "total_tokens": self.input_tokens+self.output_tokens,
                "prompt_tokens_details": {
                    "cached_tokens": self.cached_tokens,
                },
                "completion_tokens_details": {
                    "reasoning_tokens": self.reasoning_tokens,
                },
            })
        }
    }

    fn response(&self, status: &str, reason: Option<&str>) -> Value {
        json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created,
            "status": status,
            "error": null,
            "incomplete_details": if reason == Some("length") {
                json!({"reason": "max_output_tokens"})
            } else {
                Value::Null
            },
            "model": self.request.model,
            "output": self.output,
            "usage": if status == "in_progress" {
                Value::Null
            } else {
                self.usage()
            },
            "parallel_tool_calls": self.request.normalized["parallel_tool_calls"],
            "tool_choice": self.request.original.get("tool_choice")
                .cloned().unwrap_or(json!("auto")),
            "tools": self.request.original.get("tools").cloned().unwrap_or(json!([])),
            "store": self.request.store,
            "previous_response_id": self.request.original["previous_response_id"],
            "reasoning": {
                "effort": self.request.normalized["effort"],
                "summary": "auto",
            },
            "text": {
                "format": self.request.normalized["format"],
            },
            "metadata": self.request.original["metadata"],
            "instructions": self.request.original["instructions"],
            "temperature": self.request.normalized["sampling"].get("temperature")
                .cloned().unwrap_or(json!(1.0)),
            "top_p": self.request.normalized["sampling"].get("top_p")
                .cloned().unwrap_or(json!(0.95)),
            "completed_at": if status == "completed" {
                json!(super::now())
            } else {
                Value::Null
            },
            "background": false,
            "truncation": "disabled",
            "service_tier": "default",
            "user": self.request.original["user"],
            "max_output_tokens": self.request.normalized["max_tokens"],
            "thinking_token_budget": self.request.normalized["thinking_token_budget"],
        })
    }

    pub fn error_event(&mut self, message: &str) -> Value {
        if self.request.responses {
            self.event(
                "error",
                json!({
                    "code": "server_error",
                    "message": message,
                    "param": null,
                }),
            )
        } else {
            json!({
                "error": {
                    "message": message,
                    "type": "server_error",
                    "code": "server_error",
                },
            })
        }
    }

    pub fn history(&self) -> Vec<Value> {
        let mut history = self.request.normalized["messages"]
            .as_array()
            .unwrap()
            .clone();
        if self.request.responses
            && self.request.original["instructions"].is_string()
            && history.first().is_some_and(|m| {
                m["role"] == "system" && m["content"] == self.request.original["instructions"]
            })
        {
            history.remove(0);
        }
        let mut assistant = json!({
            "role": "assistant",
            "content": self.content,
            "reasoning_content": self.reasoning_text,
        });
        if !self.calls.is_empty() {
            assistant["tool_calls"] = json!(self.calls);
        }
        history.push(assistant);
        history
    }
}
