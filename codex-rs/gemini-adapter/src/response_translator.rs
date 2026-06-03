use codex_api::ResponseEvent;
use codex_protocol::error::Result;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use serde::Deserialize;
use serde_json::Value;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

static NEXT_RESPONSE_CALL_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Default)]
pub(crate) struct StreamAccumulator {
    text: String,
    calls: Vec<PendingFunctionCall>,
    usage: Option<TokenUsage>,
    finished: bool,
    emitted: bool,
}

#[derive(Debug, Clone)]
struct PendingFunctionCall {
    name: String,
    args: Value,
    thought_signature: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateContentResponse {
    #[serde(default)]
    candidates: Vec<Candidate>,
    usage_metadata: Option<UsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Candidate {
    content: Option<Content>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Content {
    #[serde(default)]
    parts: Vec<Part>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Part {
    text: Option<String>,
    function_call: Option<FunctionCall>,
    thought_signature: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FunctionCall {
    name: String,
    #[serde(default)]
    args: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageMetadata {
    prompt_token_count: Option<i64>,
    candidates_token_count: Option<i64>,
    thoughts_token_count: Option<i64>,
    cached_content_token_count: Option<i64>,
    total_token_count: Option<i64>,
}

impl StreamAccumulator {
    pub(crate) fn process_event_data(&mut self, data: &str) -> Result<Vec<ResponseEvent>> {
        let response: GenerateContentResponse = serde_json::from_str(data)?;
        if let Some(usage) = response.usage_metadata {
            self.usage = Some(usage.into());
        }
        for candidate in response.candidates {
            if let Some(content) = candidate.content {
                self.process_parts(content.parts);
            }
            if candidate.finish_reason.is_some() {
                self.finished = true;
            }
        }
        Ok(Vec::new())
    }

    fn process_parts(&mut self, parts: Vec<Part>) {
        for part in parts {
            if let Some(text) = part.text
                && !text.is_empty()
            {
                self.text.push_str(&text);
            }
            if let Some(function_call) = part.function_call {
                self.calls.push(PendingFunctionCall {
                    name: function_call.name,
                    args: function_call.args,
                    thought_signature: part.thought_signature,
                });
            } else if let Some(thought_signature) = part.thought_signature
                && let Some(last_call) = self
                    .calls
                    .iter_mut()
                    .rev()
                    .find(|call| call.thought_signature.is_none())
            {
                last_call.thought_signature = Some(thought_signature);
            }
        }
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.finished
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<ResponseEvent>> {
        if self.emitted {
            return Ok(Vec::new());
        }
        self.emitted = true;

        let mut events = Vec::new();
        if !self.text.is_empty() {
            events.push(ResponseEvent::OutputItemDone(ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: self.text.clone(),
                }],
                phase: None,
            }));
        }
        for (index, call) in self.calls.iter().enumerate() {
            let sequence = NEXT_RESPONSE_CALL_ID.fetch_add(1, Ordering::Relaxed);
            let call_id = format!("gemini-fc-{sequence}-{index}");
            events.push(ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
                id: None,
                name: call.name.clone(),
                namespace: None,
                arguments: serde_json::to_string(&call.args)?,
                call_id,
                thought_signature: call.thought_signature.clone(),
            }));
        }
        events.push(ResponseEvent::Completed {
            response_id: "gemini-response".to_string(),
            token_usage: self.usage.clone(),
            end_turn: Some(self.calls.is_empty()),
        });
        Ok(events)
    }
}

impl From<UsageMetadata> for TokenUsage {
    fn from(value: UsageMetadata) -> Self {
        let input_tokens = value.prompt_token_count.unwrap_or_default();
        let cached_input_tokens = value.cached_content_token_count.unwrap_or_default();
        let output_tokens = value.candidates_token_count.unwrap_or_default();
        let reasoning_output_tokens = value.thoughts_token_count.unwrap_or_default();
        let total_tokens = value
            .total_token_count
            .unwrap_or(input_tokens + output_tokens + reasoning_output_tokens);
        Self {
            input_tokens,
            cached_input_tokens,
            output_tokens,
            reasoning_output_tokens,
            total_tokens,
        }
    }
}

#[cfg(test)]
#[path = "response_translator_tests.rs"]
mod tests;
