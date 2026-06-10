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
use std::collections::HashMap;

use crate::GeminiPrompt;
use crate::GeminiToolChoice;
use crate::schema_sanitizer;
use crate::signature_store::SignatureStore;
use crate::tool_translator;
use crate::tool_translator::GoogleSearchGrounding;
use crate::tool_translator::Tool;

const CODEX_GEMINI_GROUNDING_ENV_VAR: &str = "CODEX_GEMINI_GROUNDING";
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) parts: Option<Vec<Part>>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) response_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) response_schema: Option<Value>,
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
    let env_value = std::env::var(CODEX_GEMINI_GROUNDING_ENV_VAR).ok();
    let google_search_grounding =
        if google_search_grounding_enabled_for_env_value(env_value.as_deref()) {
            GoogleSearchGrounding::Enabled
        } else {
            GoogleSearchGrounding::Disabled
        };
    build_generate_content_request_with_grounding(
        prompt,
        model_info,
        effort,
        google_search_grounding,
    )
}

fn google_search_grounding_enabled_for_env_value(value: Option<&str>) -> bool {
    value.is_some_and(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true"))
}

fn build_generate_content_request_with_grounding(
    prompt: &GeminiPrompt,
    model_info: &ModelInfo,
    effort: Option<ReasoningEffort>,
    google_search_grounding: GoogleSearchGrounding,
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

    let google_search_grounding = if prompt.output_schema.is_none()
        && google_search_grounding == GoogleSearchGrounding::Enabled
    {
        GoogleSearchGrounding::Enabled
    } else {
        GoogleSearchGrounding::Disabled
    };
    let tools = tool_translator::build_tools(&prompt.tools, google_search_grounding)?;
    let tool_config = tools.as_ref().map(|_| {
        let (mode, allowed_function_names) = match &prompt.tool_choice {
            GeminiToolChoice::Any {
                allowed_function_names,
            } if prompt.output_schema.is_none() => {
                ("ANY".to_string(), Some(allowed_function_names.clone()))
            }
            GeminiToolChoice::Auto | GeminiToolChoice::Any { .. } => ("AUTO".to_string(), None),
        };
        ToolConfig {
            function_calling_config: FunctionCallingConfig {
                mode,
                allowed_function_names,
            },
        }
    });
    let output_schema = prompt
        .output_schema
        .clone()
        .map(schema_sanitizer::sanitize_tool_parameters);
    let response_mime_type = output_schema
        .as_ref()
        .map(|_| "application/json".to_string());
    let system_instruction = normalize_system_instruction(&prompt.instructions);

    Ok(GenerateContentRequest {
        system_instruction: (!system_instruction.is_empty()).then(|| Content {
            role: "user".to_string(),
            parts: vec![Part {
                text: Some(system_instruction),
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
            response_mime_type,
            response_schema: output_schema,
        },
    })
}

fn normalize_system_instruction(instructions: &str) -> String {
    let mut normalized = String::new();
    let mut previous_blank = false;
    for line in instructions.trim().lines() {
        if line.trim().is_empty() {
            if !previous_blank && !normalized.is_empty() {
                normalized.push('\n');
                normalized.push('\n');
            }
            previous_blank = true;
            continue;
        }

        if !normalized.is_empty() && !normalized.ends_with('\n') {
            normalized.push('\n');
        }
        normalized.push_str(line);
        previous_blank = false;
    }
    normalized
}

fn contents_from_response_items(
    items: &[ResponseItem],
    store: &SignatureStore,
) -> Result<Vec<Content>> {
    let mut contents = Vec::new();
    let mut index = 0;
    while index < items.len() {
        match &items[index] {
            ResponseItem::Message { role, content, .. } => {
                let gemini_role = if role == "assistant" { "model" } else { "user" };
                let parts = parts_from_content_items(content)?;
                if !parts.is_empty() {
                    contents.push(Content {
                        role: gemini_role.to_string(),
                        parts,
                    });
                }
                index += 1;
            }
            ResponseItem::Reasoning {
                encrypted_content: Some(encrypted_content),
                ..
            } => {
                contents.push(Content {
                    role: "model".to_string(),
                    parts: vec![Part {
                        thought_signature: Some(encrypted_content.clone()),
                        thought: Some(true),
                        ..Part::default()
                    }],
                });
                index += 1;
            }
            ResponseItem::Reasoning {
                encrypted_content: None,
                ..
            } => {
                index += 1;
            }
            ResponseItem::FunctionCall { .. } => {
                let mut call_ids = Vec::new();
                let mut call_parts = Vec::new();
                while let Some(ResponseItem::FunctionCall {
                    name,
                    arguments,
                    call_id,
                    thought_signature,
                    ..
                }) = items.get(index)
                {
                    let args = function_call_args(arguments)?;
                    call_parts.push(Part {
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
                    });
                    call_ids.push(call_id.clone());
                    index += 1;
                }
                contents.push(Content {
                    role: "model".to_string(),
                    parts: call_parts,
                });

                let mut outputs_by_call_id = HashMap::new();
                while let Some(ResponseItem::FunctionCallOutput { call_id, output }) =
                    items.get(index)
                {
                    if !call_ids.iter().any(|candidate| candidate == call_id) {
                        break;
                    }
                    outputs_by_call_id.insert(call_id.clone(), output);
                    index += 1;
                }
                let mut response_parts = Vec::new();
                for call_id in call_ids {
                    if let Some(output) = outputs_by_call_id.remove(&call_id) {
                        let call = store.get(&call_id).ok_or_else(|| {
                            CodexErr::InvalidRequest(format!(
                                "Gemini function response is missing function call name for call_id `{call_id}`"
                            ))
                        })?;
                        let output_response = function_output_response(output)?;
                        response_parts.push(Part {
                            function_response: Some(FunctionResponse {
                                name: call.name.clone(),
                                response: output_response.response,
                                parts: output_response.parts,
                            }),
                            ..Part::default()
                        });
                    }
                }
                if !response_parts.is_empty() {
                    contents.push(Content {
                        role: "user".to_string(),
                        parts: response_parts,
                    });
                }
            }
            ResponseItem::FunctionCallOutput { call_id, output } => {
                let call = store.get(call_id).ok_or_else(|| {
                    CodexErr::InvalidRequest(format!(
                        "Gemini function response is missing function call name for call_id `{call_id}`"
                    ))
                })?;
                let output_response = function_output_response(output)?;
                contents.push(Content {
                    role: "user".to_string(),
                    parts: vec![Part {
                        function_response: Some(FunctionResponse {
                            name: call.name.clone(),
                            response: output_response.response,
                            parts: output_response.parts,
                        }),
                        ..Part::default()
                    }],
                });
                index += 1;
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

struct FunctionOutputResponse {
    response: Value,
    parts: Option<Vec<Part>>,
}

fn function_output_response(output: &FunctionCallOutputPayload) -> Result<FunctionOutputResponse> {
    let mut response = serde_json::Map::new();
    let parts = match &output.body {
        FunctionCallOutputBody::Text(text) => {
            response.insert("output".to_string(), Value::String(text.clone()));
            None
        }
        FunctionCallOutputBody::ContentItems(items) => {
            let content_response = content_items_response(items)?;
            response.insert("output".to_string(), content_response.output);
            content_response.parts
        }
    };
    if let Some(success) = output.success {
        response.insert("success".to_string(), Value::Bool(success));
    }
    Ok(FunctionOutputResponse {
        response: Value::Object(response),
        parts,
    })
}

struct ContentItemsResponse {
    output: Value,
    parts: Option<Vec<Part>>,
}

fn content_items_response(items: &[FunctionCallOutputContentItem]) -> Result<ContentItemsResponse> {
    let mut output_items = Vec::with_capacity(items.len());
    let mut image_parts = Vec::new();

    for item in items {
        match item {
            FunctionCallOutputContentItem::InputImage { image_url, .. } => {
                image_parts.push(image_part(image_url)?);
            }
            FunctionCallOutputContentItem::InputText { .. }
            | FunctionCallOutputContentItem::EncryptedContent { .. } => {
                output_items.push(item.clone());
            }
        }
    }

    let output = if image_parts.is_empty() {
        serde_json::to_value(items)?
    } else if output_items.is_empty() {
        Value::String("Image output attached.".to_string())
    } else {
        serde_json::to_value(output_items)?
    };
    let parts = (!image_parts.is_empty()).then_some(image_parts);

    Ok(ContentItemsResponse { output, parts })
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
