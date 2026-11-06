use crate::serving::events::Output;
use serde_json::{Value, json};

impl Output {
    pub(super) fn reasoning_delta(&mut self, text: String, events: &mut Vec<Value>) {
        if text.is_empty() {
            return;
        }
        self.reasoning_text.push_str(&text);
        let index = if let Some(i) = self.reasoning {
            i
        } else {
            let i = self.output.len();
            let item = json!({
                "id": format!("rs_{}",self.id),
                "type": "reasoning",
                "summary": [],
            });
            self.output.push(item.clone());
            self.reasoning = Some(i);
            events.push(self.event(
                "response.output_item.added",
                json!({
                    "output_index": i,
                    "item": item,
                }),
            ));
            events.push(self.event(
                "response.reasoning_summary_part.added",
                json!({
                    "item_id": item["id"],
                    "output_index": i,
                    "summary_index": 0,
                    "part": {
                        "type": "summary_text",
                        "text": "",
                    },
                }),
            ));
            self.output[i]["summary"] = json!([
                {
                    "type": "summary_text",
                    "text": "",
                },
            ]);
            i
        };
        if let Value::String(value) = &mut self.output[index]["summary"][0]["text"] {
            value.push_str(&text);
        }
        events.push(self.event(
            "response.reasoning_summary_text.delta",
            json!({
                "item_id": self.output[index]["id"],
                "output_index": index,
                "summary_index": 0,
                "delta": text,
            }),
        ));
    }

    pub(super) fn text_delta(&mut self, text: String, events: &mut Vec<Value>) {
        if text.is_empty() {
            return;
        }
        self.content.push_str(&text);
        let index = if let Some(i) = self.message {
            i
        } else {
            let i = self.output.len();
            let item = json!({
                "id": format!("msg_{}",self.id),
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            });
            self.output.push(item.clone());
            self.message = Some(i);
            events.push(self.event(
                "response.output_item.added",
                json!({
                    "output_index": i,
                    "item": item,
                }),
            ));
            let part = json!({
                "type": "output_text",
                "text": "",
                "annotations": [],
                "logprobs": [],
            });
            events.push(self.event(
                "response.content_part.added",
                json!({
                    "item_id": item["id"],
                    "output_index": i,
                    "content_index": 0,
                    "part": part,
                }),
            ));
            self.output[i]["content"] = json!([part,]);
            i
        };
        if let Value::String(value) = &mut self.output[index]["content"][0]["text"] {
            value.push_str(&text);
        }
        events.push(self.event(
            "response.output_text.delta",
            json!({
                "item_id": self.output[index]["id"],
                "output_index": index,
                "content_index": 0,
                "delta": text,
                "logprobs": [],
            }),
        ));
    }

    pub(super) fn tool_delta(
        &mut self,
        index: usize,
        name: String,
        arguments: String,
        events: &mut Vec<Value>,
    ) {
        let call_id = format!("call_{}_{}", self.id, index);
        self.calls.push(json!({
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }));
        let i = self.output.len();
        let item = json!({
            "id": format!("fc_{}_{}",self.id,index),
            "type": "function_call",
            "status": "in_progress",
            "call_id": call_id,
            "name": name,
            "arguments": "",
        });
        self.output.push(item.clone());
        events.push(self.event(
            "response.output_item.added",
            json!({
                "output_index": i,
                "item": item,
            }),
        ));
        events.push(self.event(
            "response.function_call_arguments.delta",
            json!({
                "item_id": item["id"],
                "output_index": i,
                "delta": arguments,
            }),
        ));
        self.output[i]["arguments"] = json!(arguments);
    }
}
