use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ReasoningEffort;
use serde::Serialize;
use serde_json::Value;

use crate::GeminiPrompt;
use crate::signature_store::SignatureStore;
use crate::tool_translator;
use crate::tool_translator::Tool;

const GEMINI_MAX_OUTPUT_TOKENS: i64 = 64_000;

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GenerateContentRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) system_instruction: Option<Content>,
    pub(crate) contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_config: Option<ToolConfig>,
    pub(crate) generation_config: GenerationConfig,
}

#[derive(Debug, Serialize, Clone, PartialEq)]
pub(crate) struct Content {
    pub(crate) role: String,
    pub(crate) parts: Vec<Part>,
}

#[derive(Debug, Serialize, Default, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) function_response: Option<FunctionResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) inline_data: Option<InlineData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) thought_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) thought: Option<bool>,
}

#[derive(Debug, Serialize, Clone, PartialEq)]
pub(crate) struct FunctionCall {
    pub(crate) name: String,
    pub(crate) args: Value,
}

#[derive(Debug, Serialize, Clone, PartialEq)]
pub(crate) struct FunctionResponse {
    pub(crate) name: String,
    pub(crate) response: Value,
}

#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InlineData {
    pub(crate) mime_type: String,
    pub(crate) data: String,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ToolConfig {
    pub(crate) function_calling_config: FunctionCallingConfig,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FunctionCallingConfig {
    pub(crate) mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) allowed_function_names: Option<Vec<String>>,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GenerationConfig {
    pub(crate) temperature: f32,
    pub(crate) max_output_tokens: i64,
    pub(crate) thinking_config: ThinkingConfig,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ThinkingConfig {
    pub(crate) thinking_level: String,
    pub(crate) include_thoughts: bool,
}

pub(crate) fn build_generate_content_request(
    prompt: &GeminiPrompt,
    model_info: &ModelInfo,
    effort: Option<ReasoningEffort>,
) -> Result<GenerateContentRequest> {
    let mut store = SignatureStore::default();
    for item in &prompt.input {
        if let ResponseItem::FunctionCall {
            call_id,
            name,
            thought_signature,
            ..
        } = item
        {
            store.record(call_id.clone(), name.clone(), thought_signature.clone());
        }
    }

    let tools = tool_translator::build_tools(&prompt.tools)?;
    let tool_config = tools.as_ref().map(|_| ToolConfig {
        function_calling_config: FunctionCallingConfig {
            mode: "AUTO".to_string(),
            allowed_function_names: None,
        },
    });

    Ok(GenerateContentRequest {
        system_instruction: (!prompt.instructions.trim().is_empty()).then(|| Content {
            role: "user".to_string(),
            parts: vec![Part {
                text: Some(prompt.instructions.clone()),
                ..Part::default()
            }],
        }),
        contents: contents_from_response_items(&prompt.input, &store)?,
        tools,
        tool_config,
        generation_config: GenerationConfig {
            temperature: 1.0,
            max_output_tokens: GEMINI_MAX_OUTPUT_TOKENS,
            thinking_config: ThinkingConfig {
                thinking_level: thinking_level(effort.or(model_info.default_reasoning_level)),
                include_thoughts: true,
            },
        },
    })
}

fn contents_from_response_items(
    items: &[ResponseItem],
    store: &SignatureStore,
) -> Result<Vec<Content>> {
    let mut contents = Vec::new();
    for item in items {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let gemini_role = if role == "assistant" { "model" } else { "user" };
                let parts = parts_from_content_items(content)?;
                if !parts.is_empty() {
                    contents.push(Content {
                        role: gemini_role.to_string(),
                        parts,
                    });
                }
            }
            ResponseItem::Reasoning {
                encrypted_content: Some(encrypted_content),
                ..
            } => contents.push(Content {
                role: "model".to_string(),
                parts: vec![Part {
                    thought_signature: Some(encrypted_content.clone()),
                    thought: Some(true),
                    ..Part::default()
                }],
            }),
            ResponseItem::Reasoning {
                encrypted_content: None,
                ..
            } => {}
            ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                thought_signature,
                ..
            } => {
                let args = function_call_args(arguments)?;
                contents.push(Content {
                    role: "model".to_string(),
                    parts: vec![Part {
                        function_call: Some(FunctionCall {
                            name: name.clone(),
                            args,
                        }),
                        thought_signature: thought_signature.clone().or_else(|| {
                            store
                                .get(call_id)
                                .and_then(|call| call.thought_signature.clone())
                        }),
                        ..Part::default()
                    }],
                });
            }
            ResponseItem::FunctionCallOutput { call_id, output } => {
                let call = store.get(call_id).ok_or_else(|| {
                    CodexErr::InvalidRequest(format!(
                        "Gemini function response is missing function call name for call_id `{call_id}`"
                    ))
                })?;
                contents.push(Content {
                    role: "user".to_string(),
                    parts: vec![Part {
                        function_response: Some(FunctionResponse {
                            name: call.name.clone(),
                            response: function_output_response(output)?,
                        }),
                        ..Part::default()
                    }],
                });
            }
            ResponseItem::LocalShellCall { .. }
            | ResponseItem::ToolSearchCall { .. }
            | ResponseItem::ToolSearchOutput { .. }
            | ResponseItem::CustomToolCall { .. }
            | ResponseItem::CustomToolCallOutput { .. }
            | ResponseItem::WebSearchCall { .. }
            | ResponseItem::ImageGenerationCall { .. }
            | ResponseItem::Compaction { .. }
            | ResponseItem::CompactionTrigger
            | ResponseItem::Other
            | ResponseItem::ContextCompaction { .. } => {
                return Err(CodexErr::UnsupportedOperation(
                    "Gemini Phase 1 cannot translate this response item".to_string(),
                ));
            }
        }
    }
    Ok(contents)
}

fn parts_from_content_items(items: &[ContentItem]) -> Result<Vec<Part>> {
    let mut parts = Vec::new();
    for item in items {
        match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                if !text.is_empty() {
                    parts.push(Part {
                        text: Some(text.clone()),
                        ..Part::default()
                    });
                }
            }
            ContentItem::InputImage { image_url, .. } => {
                parts.push(image_part(image_url)?);
            }
        }
    }
    Ok(parts)
}

fn image_part(image_url: &str) -> Result<Part> {
    let Some(data_url) = image_url.strip_prefix("data:") else {
        return Err(CodexErr::UnsupportedOperation(
            "Gemini Phase 1 only supports data URL input images".to_string(),
        ));
    };
    let Some((mime_type, data)) = data_url.split_once(";base64,") else {
        return Err(CodexErr::InvalidRequest(
            "Gemini image data URL must be base64 encoded".to_string(),
        ));
    };
    Ok(Part {
        inline_data: Some(InlineData {
            mime_type: mime_type.to_string(),
            data: data.to_string(),
        }),
        ..Part::default()
    })
}

fn function_output_response(output: &FunctionCallOutputPayload) -> Result<Value> {
    let mut response = serde_json::Map::new();
    match &output.body {
        FunctionCallOutputBody::Text(text) => {
            response.insert("output".to_string(), Value::String(text.clone()));
        }
        FunctionCallOutputBody::ContentItems(items) => {
            response.insert("output".to_string(), content_items_response(items)?);
        }
    }
    if let Some(success) = output.success {
        response.insert("success".to_string(), Value::Bool(success));
    }
    Ok(Value::Object(response))
}

fn function_call_args(arguments: &str) -> Result<Value> {
    let args: Value = serde_json::from_str(arguments)?;
    if args.is_object() {
        Ok(args)
    } else {
        Err(CodexErr::InvalidRequest(
            "Gemini functionCall args must be a JSON object".to_string(),
        ))
    }
}

fn content_items_response(items: &[FunctionCallOutputContentItem]) -> Result<Value> {
    serde_json::to_value(items).map_err(Into::into)
}

fn thinking_level(effort: Option<ReasoningEffort>) -> String {
    match effort.unwrap_or(ReasoningEffort::High) {
        ReasoningEffort::None | ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High | ReasoningEffort::XHigh => "high",
    }
    .to_string()
}

#[cfg(test)]
#[path = "request_translator_tests.rs"]
mod tests;
