use super::{parser::Delta, request::Request};
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
        json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,"model":self.request.model,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]})
    }
    pub fn start(&mut self) -> Vec<Value> {
        if !self.request.responses {
            return vec![self.chat_chunk(json!({"role":"assistant","content":""}), Value::Null)];
        }
        let response = self.response("in_progress", None);
        vec![
            self.event("response.created", json!({"response":response})),
            self.event("response.in_progress", json!({"response":response})),
        ]
    }
    pub fn delta(&mut self, delta: Delta) -> Vec<Value> {
        let mut events = Vec::new();
        if !self.request.responses {
            let wire = match delta {
                Delta::Text(text) => {
                    self.content.push_str(&text);
                    json!({"content":text})
                }
                Delta::Reasoning(text) => {
                    self.reasoning_text.push_str(&text);
                    json!({"reasoning_content":text})
                }
                Delta::Tool {
                    index,
                    name,
                    arguments,
                } => {
                    let call = json!({"id":format!("call_{}_{}",self.id,index),"type":"function","function":{"name":name,"arguments":arguments}});
                    self.calls.push(call.clone());
                    let mut c = call;
                    c["index"] = json!(index);
                    json!({"tool_calls":[c]})
                }
            };
            return vec![self.chat_chunk(wire, Value::Null)];
        }
        match delta {
            Delta::Reasoning(text) => {
                if text.is_empty() {
                    return events;
                }
                self.reasoning_text.push_str(&text);
                let index = if let Some(i) = self.reasoning {
                    i
                } else {
                    let i = self.output.len();
                    let item =
                        json!({"id":format!("rs_{}",self.id),"type":"reasoning","summary":[]});
                    self.output.push(item.clone());
                    self.reasoning = Some(i);
                    events.push(self.event(
                        "response.output_item.added",
                        json!({"output_index":i,"item":item}),
                    ));
                    events.push(self.event("response.reasoning_summary_part.added", json!({"item_id":item["id"],"output_index":i,"summary_index":0,"part":{"type":"summary_text","text":""}})));
                    self.output[i]["summary"] = json!([{"type":"summary_text","text":""}]);
                    i
                };
                if let Value::String(value) = &mut self.output[index]["summary"][0]["text"] {
                    value.push_str(&text);
                }
                events.push(self.event("response.reasoning_summary_text.delta", json!({"item_id":self.output[index]["id"],"output_index":index,"summary_index":0,"delta":text})));
            }
            Delta::Text(text) => {
                if text.is_empty() {
                    return events;
                }
                self.content.push_str(&text);
                let index = if let Some(i) = self.message {
                    i
                } else {
                    let i = self.output.len();
                    let item = json!({"id":format!("msg_{}",self.id),"type":"message","status":"in_progress","role":"assistant","content":[]});
                    self.output.push(item.clone());
                    self.message = Some(i);
                    events.push(self.event(
                        "response.output_item.added",
                        json!({"output_index":i,"item":item}),
                    ));
                    let part =
                        json!({"type":"output_text","text":"","annotations":[],"logprobs":[]});
                    events.push(self.event("response.content_part.added", json!({"item_id":item["id"],"output_index":i,"content_index":0,"part":part})));
                    self.output[i]["content"] = json!([part]);
                    i
                };
                if let Value::String(value) = &mut self.output[index]["content"][0]["text"] {
                    value.push_str(&text);
                }
                events.push(self.event("response.output_text.delta", json!({"item_id":self.output[index]["id"],"output_index":index,"content_index":0,"delta":text,"logprobs":[]})));
            }
            Delta::Tool {
                index,
                name,
                arguments,
            } => {
                let call_id = format!("call_{}_{}", self.id, index);
                self.calls.push(json!({"id":call_id,"type":"function","function":{"name":name,"arguments":arguments}}));
                let i = self.output.len();
                let item = json!({"id":format!("fc_{}_{}",self.id,index),"type":"function_call","status":"in_progress","call_id":call_id,"name":name,"arguments":""});
                self.output.push(item.clone());
                events.push(self.event(
                    "response.output_item.added",
                    json!({"output_index":i,"item":item}),
                ));
                events.push(self.event(
                    "response.function_call_arguments.delta",
                    json!({"item_id":item["id"],"output_index":i,"delta":arguments}),
                ));
                self.output[i]["arguments"] = json!(arguments);
            }
        }
        events
    }
    pub fn finish(&mut self, reason: &str) -> (Value, Vec<Value>) {
        if !self.request.responses {
            let finish = if reason == "length" {
                "length"
            } else if !self.calls.is_empty() {
                "tool_calls"
            } else {
                "stop"
            };
            let mut events = vec![self.chat_chunk(json!({}), json!(finish))];
            if self.request.include_usage {
                events.push(json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,"model":self.request.model,"choices":[],"usage":self.usage()}));
            }
            let mut message = json!({"role":"assistant","content":if self.content.is_empty() {Value::Null} else {json!(self.content)}});
            if !self.reasoning_text.is_empty() {
                message["reasoning_content"] = json!(self.reasoning_text);
            }
            if !self.calls.is_empty() {
                message["tool_calls"] = json!(self.calls);
            }
            let result = json!({"id":self.id,"object":"chat.completion","created":self.created,"model":self.request.model,"choices":[{"index":0,"message":message,"finish_reason":finish}],"usage":self.usage()});
            return (result, events);
        }
        let mut events = Vec::new();
        for i in 0..self.output.len() {
            let item = self.output[i].clone();
            match item["type"].as_str().unwrap() {
                "reasoning" => {
                    events.push(self.event("response.reasoning_summary_text.done", json!({"item_id":item["id"],"output_index":i,"summary_index":0,"text":self.reasoning_text})));
                    events.push(self.event("response.reasoning_summary_part.done", json!({"item_id":item["id"],"output_index":i,"summary_index":0,"part":item["summary"][0]})));
                }
                "message" => {
                    events.push(self.event("response.output_text.done", json!({"item_id":item["id"],"output_index":i,"content_index":0,"text":self.content,"logprobs":[]})));
                    events.push(self.event("response.content_part.done", json!({"item_id":item["id"],"output_index":i,"content_index":0,"part":item["content"][0]})));
                    self.output[i]["status"] = json!(if reason == "length" {
                        "incomplete"
                    } else {
                        "completed"
                    });
                }
                "function_call" => {
                    events.push(self.event("response.function_call_arguments.done", json!({"item_id":item["id"],"output_index":i,"name":item["name"],"arguments":item["arguments"]})));
                    self.output[i]["status"] = json!("completed");
                }
                _ => unreachable!(),
            }
            events.push(self.event(
                "response.output_item.done",
                json!({"output_index":i,"item":self.output[i]}),
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
            json!({"response":response}),
        ));
        (response, events)
    }
    fn usage(&self) -> Value {
        if self.request.responses {
            json!({"input_tokens":self.input_tokens,"output_tokens":self.output_tokens,"total_tokens":self.input_tokens+self.output_tokens,"input_tokens_details":{"cached_tokens":self.cached_tokens},"output_tokens_details":{"reasoning_tokens":self.reasoning_tokens}})
        } else {
            json!({"prompt_tokens":self.input_tokens,"completion_tokens":self.output_tokens,"total_tokens":self.input_tokens+self.output_tokens,"prompt_tokens_details":{"cached_tokens":self.cached_tokens},"completion_tokens_details":{"reasoning_tokens":self.reasoning_tokens}})
        }
    }
    fn response(&self, status: &str, reason: Option<&str>) -> Value {
        json!({"id":self.id,"object":"response","created_at":self.created,"status":status,"error":null,"incomplete_details":if reason == Some("length") {json!({"reason":"max_output_tokens"})} else {Value::Null},"model":self.request.model,"output":self.output,"usage":if status == "in_progress" {Value::Null} else {self.usage()},"parallel_tool_calls":self.request.normalized["parallel_tool_calls"],"tool_choice":self.request.original.get("tool_choice").cloned().unwrap_or(json!("auto")),"tools":self.request.original.get("tools").cloned().unwrap_or(json!([])),"store":self.request.store,"previous_response_id":self.request.original["previous_response_id"],"reasoning":{"effort":self.request.normalized["effort"],"summary":"auto"},"text":{"format":self.request.normalized["format"]},"metadata":self.request.original["metadata"],"instructions":self.request.original["instructions"],"temperature":self.request.normalized["sampling"].get("temperature").cloned().unwrap_or(json!(1.0)),"top_p":self.request.normalized["sampling"].get("top_p").cloned().unwrap_or(json!(0.95)),"completed_at":if status == "completed" {json!(super::now())} else {Value::Null},"background":false,"truncation":"disabled","service_tier":"default","user":self.request.original["user"],"max_output_tokens":self.request.normalized["max_tokens"]})
    }
    pub fn error_event(&mut self, message: &str) -> Value {
        if self.request.responses {
            self.event(
                "error",
                json!({"code":"server_error","message":message,"param":null}),
            )
        } else {
            json!({"error":{"message":message,"type":"server_error","code":"server_error"}})
        }
    }
    pub fn history(&self) -> Vec<Value> {
        let mut history = self.request.normalized["messages"]
            .as_array()
            .unwrap()
            .clone();
        // Responses instructions apply only to this response, unlike input messages.
        if self.request.responses
            && self.request.original["instructions"].is_string()
            && history.first().is_some_and(|m| {
                m["role"] == "system" && m["content"] == self.request.original["instructions"]
            })
        {
            history.remove(0);
        }
        let mut assistant = json!({"role":"assistant","content":self.content,"reasoning_content":self.reasoning_text});
        if !self.calls.is_empty() {
            assistant["tool_calls"] = json!(self.calls);
        }
        history.push(assistant);
        history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stored_history_drops_only_top_level_instructions() {
        let request = super::super::request::normalize(
            json!({"model":"m","instructions":"one turn only","input":"hello"}),
            true,
            "m",
            None,
        )
        .unwrap();
        let mut output = Output::new("resp_test".to_owned(), request);
        output.delta(Delta::Reasoning("thinking".to_owned()));
        output.delta(Delta::Text("answer".to_owned()));
        let history = output.history();
        assert_eq!(history[0]["role"], "user");
        assert_eq!(history[1]["reasoning_content"], "thinking");
        let followup = super::super::request::normalize(
            json!({"model":"m","input":"continue"}),
            true,
            "m",
            Some(history),
        )
        .unwrap();
        assert!(
            followup.normalized["messages"]
                .as_array()
                .unwrap()
                .iter()
                .all(|m| m["role"] != "system")
        );
    }
}
