use crate::serving::events::Output;
use serde_json::{Value, json};

impl Output {
    pub(super) fn finish_item(
        &mut self,
        i: usize,
        item: &Value,
        reason: &str,
        events: &mut Vec<Value>,
    ) {
        match item["type"].as_str().unwrap() {
            "reasoning" => self.finish_reasoning(i, item, events),
            "message" => self.finish_message(i, item, reason, events),
            "function_call" => self.finish_tool(i, item, events),
            _ => unreachable!(),
        }
    }

    fn finish_reasoning(&mut self, i: usize, item: &Value, events: &mut Vec<Value>) {
        events.push(self.event(
            "response.reasoning_summary_text.done",
            json!({
                "item_id": item["id"],
                "output_index": i,
                "summary_index": 0,
                "text": self.reasoning_text,
            }),
        ));
        events.push(self.event(
            "response.reasoning_summary_part.done",
            json!({
                "item_id": item["id"],
                "output_index": i,
                "summary_index": 0,
                "part": item["summary"][0],
            }),
        ));
    }

    fn finish_message(&mut self, i: usize, item: &Value, reason: &str, events: &mut Vec<Value>) {
        events.push(self.event(
            "response.output_text.done",
            json!({
                "item_id": item["id"],
                "output_index": i,
                "content_index": 0,
                "text": self.content,
                "logprobs": [],
            }),
        ));
        events.push(self.event(
            "response.content_part.done",
            json!({
                "item_id": item["id"],
                "output_index": i,
                "content_index": 0,
                "part": item["content"][0],
            }),
        ));
        self.output[i]["status"] = json!(if reason == "length" {
            "incomplete"
        } else {
            "completed"
        });
    }

    fn finish_tool(&mut self, i: usize, item: &Value, events: &mut Vec<Value>) {
        events.push(self.event(
            "response.function_call_arguments.done",
            json!({
                "item_id": item["id"],
                "output_index": i,
                "name": item["name"],
                "arguments": item["arguments"],
            }),
        ));
        self.output[i]["status"] = json!("completed");
    }
}
