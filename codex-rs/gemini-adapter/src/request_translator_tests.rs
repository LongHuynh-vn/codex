use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;
use codex_protocol::models::WebSearchAction;
use codex_protocol::protocol::GeminiSearchMode;
use codex_tools::AdditionalProperties;
use codex_tools::JsonSchema;
use codex_tools::JsonSchemaPrimitiveType;
use codex_tools::JsonSchemaType;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;

use super::*;
use crate::GeminiPrompt;
use crate::GeminiThoughtSummaryDisplay;
use crate::GeminiToolChoice;
use crate::model_config::gemini_model_catalog;

#[test]
fn builds_native_request_with_thinking_and_signature_replay() {
    let mut catalog = gemini_model_catalog();
    let model_info = catalog.models.remove(0);
    let prompt = GeminiPrompt {
        instructions: "You are Codex.".to_string(),
        input: vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "run the step tool".to_string(),
                }],
                phase: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "record_step".to_string(),
                namespace: None,
                arguments: r#"{"step":1}"#.to_string(),
                call_id: "call-1".to_string(),
                thought_signature: Some("sig-1".to_string()),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call-1".to_string(),
                output: FunctionCallOutputPayload::from_text("ok".to_string()),
            },
        ],
        tools: vec![ToolSpec::Function(ResponsesApiTool {
            name: "record_step".to_string(),
            description: "Record a step.".to_string(),
            strict: false,
            defer_loading: None,
            parameters: Default::default(),
            output_schema: None,
        })],
        output_schema: None,
        gemini_search_mode: None,
        tool_choice: GeminiToolChoice::Auto,
        thought_summary_display: GeminiThoughtSummaryDisplay::Hidden,
    };

    let request =
        build_generate_content_request(&prompt, &model_info, Some(ReasoningEffort::High)).unwrap();
    let value = serde_json::to_value(request).unwrap();

    assert_eq!(value["generationConfig"]["temperature"], json!(1.0));
    assert_eq!(
        value["generationConfig"]["thinkingConfig"],
        json!({"thinkingLevel": "high", "includeThoughts": true})
    );
    assert!(value["generationConfig"].get("thinkingBudget").is_none());
    assert_eq!(
        value["contents"][1]["parts"][0]["thoughtSignature"],
        json!("sig-1")
    );
    assert_eq!(
        value["contents"][2]["parts"][0]["functionResponse"],
        json!({"name": "record_step", "response": {"output": "ok"}})
    );
}

#[test]
fn parses_google_search_grounding_env_toggle() {
    assert!(!google_search_grounding_enabled_for_env_value(None));
    assert!(!google_search_grounding_enabled_for_env_value(Some("")));
    assert!(!google_search_grounding_enabled_for_env_value(Some("0")));
    assert!(!google_search_grounding_enabled_for_env_value(Some("yes")));
    assert!(google_search_grounding_enabled_for_env_value(Some("1")));
    assert!(google_search_grounding_enabled_for_env_value(Some("true")));
    assert!(google_search_grounding_enabled_for_env_value(Some(
        " TRUE "
    )));
}

#[test]
fn gemini_search_mode_controls_google_search_grounding() {
    let cases = [
        (GeminiSearchMode::Tavily, GoogleSearchGrounding::Disabled),
        (GeminiSearchMode::Grounding, GoogleSearchGrounding::Enabled),
        (GeminiSearchMode::Hybrid, GoogleSearchGrounding::Enabled),
        (GeminiSearchMode::Off, GoogleSearchGrounding::Disabled),
    ];

    for (mode, expected) in cases {
        assert_eq!(google_search_grounding_for_mode(Some(mode)), expected);
    }
}

#[test]
fn google_search_grounding_disabled_omits_google_search_tool() {
    let value =
        request_value_with_grounding(vec![weather_tool()], None, GoogleSearchGrounding::Disabled);
    let tools = value["tools"].as_array().expect("tools should serialize");

    assert_eq!(tools.len(), 1);
    assert!(tools[0].get("functionDeclarations").is_some());
    assert!(tools.iter().all(|tool| tool.get("googleSearch").is_none()));
}

#[test]
fn google_search_grounding_appends_after_function_declarations() {
    let value =
        request_value_with_grounding(vec![weather_tool()], None, GoogleSearchGrounding::Enabled);
    let tools = value["tools"].as_array().expect("tools should serialize");

    assert_eq!(tools.len(), 2);
    assert_eq!(
        tools[0]["functionDeclarations"][0]["name"],
        json!("get_weather")
    );
    assert_eq!(tools[1], json!({"googleSearch": {}}));
}

#[test]
fn google_search_grounding_does_not_emit_google_search_only_tools() {
    let value = request_value_with_grounding(Vec::new(), None, GoogleSearchGrounding::Enabled);

    assert!(value.get("tools").is_none());
    assert!(value.get("toolConfig").is_none());
}

#[test]
fn google_search_grounding_is_disabled_for_structured_output() {
    let value = request_value_with_grounding(
        vec![weather_tool()],
        Some(json!({
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"}
            },
            "required": ["ok"]
        })),
        GoogleSearchGrounding::Enabled,
    );
    let tools = value["tools"].as_array().expect("tools should serialize");

    assert_eq!(tools.len(), 1);
    assert!(tools[0].get("functionDeclarations").is_some());
    assert!(tools.iter().all(|tool| tool.get("googleSearch").is_none()));
    assert_eq!(
        value["generationConfig"]["responseMimeType"],
        json!("application/json")
    );
}

#[test]
fn sanitizes_known_bad_tool_schema_for_gemini() {
    let mut catalog = gemini_model_catalog();
    let model_info = catalog.models.remove(0);
    let prompt = GeminiPrompt {
        instructions: String::new(),
        input: vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "Use the tool if needed.".to_string(),
            }],
            phase: None,
        }],
        tools: vec![ToolSpec::Function(ResponsesApiTool {
            name: "complex_tool".to_string(),
            description: "A deliberately complex schema.".to_string(),
            strict: false,
            defer_loading: None,
            parameters: JsonSchema::object(
                BTreeMap::from([
                    (
                        "city".to_string(),
                        JsonSchema::string(Some("City to inspect.".to_string())),
                    ),
                    (
                        "metadata".to_string(),
                        JsonSchema {
                            schema_ref: Some("#/$defs/Metadata".to_string()),
                            schema_type: Some(JsonSchemaType::Single(
                                JsonSchemaPrimitiveType::Object,
                            )),
                            properties: Some(BTreeMap::from([(
                                "nested".to_string(),
                                JsonSchema::object(
                                    BTreeMap::from([(
                                        "value".to_string(),
                                        JsonSchema::integer(Some("Nested value.".to_string())),
                                    )]),
                                    Some(vec!["value".to_string()]),
                                    Some(AdditionalProperties::Boolean(false)),
                                ),
                            )])),
                            additional_properties: Some(AdditionalProperties::Boolean(false)),
                            ..Default::default()
                        },
                    ),
                ]),
                Some(vec!["city".to_string()]),
                Some(AdditionalProperties::Boolean(false)),
            ),
            output_schema: None,
        })],
        output_schema: None,
        gemini_search_mode: None,
        tool_choice: GeminiToolChoice::Auto,
        thought_summary_display: GeminiThoughtSummaryDisplay::Hidden,
    };

    let request =
        build_generate_content_request(&prompt, &model_info, Some(ReasoningEffort::Low)).unwrap();
    let value = serde_json::to_value(request).unwrap();
    let parameters = &value["tools"][0]["functionDeclarations"][0]["parameters"];

    assert!(parameters.get("additionalProperties").is_none());
    assert!(
        parameters["properties"]["metadata"]
            .get("additionalProperties")
            .is_none()
    );
    assert!(parameters["properties"]["metadata"].get("$ref").is_none());
    assert!(
        parameters["properties"]["metadata"]["properties"]["nested"]
            .get("additionalProperties")
            .is_none()
    );
    assert_eq!(
        parameters["properties"]["metadata"]["properties"]["nested"]["properties"]["value"]["type"],
        json!("integer")
    );
    assert!(value.get("parallelToolCalls").is_none());
    assert!(value.get("parallel_tool_calls").is_none());
}

#[test]
fn serializes_parallel_function_calls_before_results_by_call_id() {
    let mut catalog = gemini_model_catalog();
    let model_info = catalog.models.remove(0);
    let prompt = GeminiPrompt {
        instructions: "You are Codex.".to_string(),
        input: vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "Get weather for Paris and London.".to_string(),
                }],
                phase: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "get_weather".to_string(),
                namespace: None,
                arguments: r#"{"city":"Paris"}"#.to_string(),
                call_id: "call-paris".to_string(),
                thought_signature: Some("sig-paris".to_string()),
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "get_weather".to_string(),
                namespace: None,
                arguments: r#"{"city":"London"}"#.to_string(),
                call_id: "call-london".to_string(),
                thought_signature: None,
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call-london".to_string(),
                output: FunctionCallOutputPayload::from_text("weather London".to_string()),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call-paris".to_string(),
                output: FunctionCallOutputPayload::from_text("weather Paris".to_string()),
            },
        ],
        tools: vec![weather_tool()],
        output_schema: None,
        gemini_search_mode: None,
        tool_choice: GeminiToolChoice::Auto,
        thought_summary_display: GeminiThoughtSummaryDisplay::Hidden,
    };

    let request =
        build_generate_content_request(&prompt, &model_info, Some(ReasoningEffort::Low)).unwrap();
    let value = serde_json::to_value(request).unwrap();

    assert_eq!(
        value["contents"][1]["parts"],
        json!([
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {"city": "Paris"}
                },
                "thoughtSignature": "sig-paris"
            },
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {"city": "London"}
                }
            }
        ])
    );
    assert_eq!(
        value["contents"][2]["parts"],
        json!([
            {
                "functionResponse": {
                    "name": "get_weather",
                    "response": {"output": "weather Paris"}
                }
            },
            {
                "functionResponse": {
                    "name": "get_weather",
                    "response": {"output": "weather London"}
                }
            }
        ])
    );
}

#[test]
fn serializes_function_output_image_as_nested_function_response_part() {
    let output = FunctionCallOutputPayload {
        body: FunctionCallOutputBody::ContentItems(vec![
            FunctionCallOutputContentItem::InputImage {
                image_url: "data:image/png;base64,AAA".to_string(),
                detail: None,
            },
        ]),
        success: Some(true),
    };

    let value = request_value_for_function_output(output);
    let function_response = &value["contents"][2]["parts"][0]["functionResponse"];

    assert_eq!(
        function_response,
        &json!({
            "name": "get_weather",
            "response": {
                "output": "Image output attached.",
                "success": true
            },
            "parts": [
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": "AAA"
                    }
                }
            ]
        })
    );
    assert!(!function_response["response"].to_string().contains("AAA"));
}

#[test]
fn serializes_text_content_items_without_nested_parts() {
    let output = FunctionCallOutputPayload {
        body: FunctionCallOutputBody::ContentItems(vec![
            FunctionCallOutputContentItem::InputText {
                text: "weather Paris".to_string(),
            },
        ]),
        success: None,
    };

    let value = request_value_for_function_output(output);
    let function_response = &value["contents"][2]["parts"][0]["functionResponse"];

    assert_eq!(
        function_response,
        &json!({
            "name": "get_weather",
            "response": {
                "output": [
                    {
                        "type": "input_text",
                        "text": "weather Paris"
                    }
                ]
            }
        })
    );
}

#[test]
fn serializes_mixed_content_items_with_images_nested() {
    let output = FunctionCallOutputPayload {
        body: FunctionCallOutputBody::ContentItems(vec![
            FunctionCallOutputContentItem::InputText {
                text: "chart follows".to_string(),
            },
            FunctionCallOutputContentItem::InputImage {
                image_url: "data:image/jpeg;base64,BBB".to_string(),
                detail: None,
            },
            FunctionCallOutputContentItem::EncryptedContent {
                encrypted_content: "enc-1".to_string(),
            },
        ]),
        success: Some(true),
    };

    let value = request_value_for_function_output(output);
    let function_response = &value["contents"][2]["parts"][0]["functionResponse"];

    assert_eq!(
        function_response,
        &json!({
            "name": "get_weather",
            "response": {
                "output": [
                    {
                        "type": "input_text",
                        "text": "chart follows"
                    },
                    {
                        "type": "encrypted_content",
                        "encrypted_content": "enc-1"
                    }
                ],
                "success": true
            },
            "parts": [
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": "BBB"
                    }
                }
            ]
        })
    );
}

#[test]
fn serializes_parallel_function_output_images_inside_matching_response_part() {
    let mut catalog = gemini_model_catalog();
    let model_info = catalog.models.remove(0);
    let prompt = GeminiPrompt {
        instructions: "You are Codex.".to_string(),
        input: vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "Get weather for Paris and London.".to_string(),
                }],
                phase: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "get_weather".to_string(),
                namespace: None,
                arguments: r#"{"city":"Paris"}"#.to_string(),
                call_id: "call-paris".to_string(),
                thought_signature: Some("sig-paris".to_string()),
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "get_weather".to_string(),
                namespace: None,
                arguments: r#"{"city":"London"}"#.to_string(),
                call_id: "call-london".to_string(),
                thought_signature: None,
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call-london".to_string(),
                output: FunctionCallOutputPayload::from_text("weather London".to_string()),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call-paris".to_string(),
                output: FunctionCallOutputPayload {
                    body: FunctionCallOutputBody::ContentItems(vec![
                        FunctionCallOutputContentItem::InputImage {
                            image_url: "data:image/webp;base64,CCC".to_string(),
                            detail: None,
                        },
                    ]),
                    success: Some(true),
                },
            },
        ],
        tools: vec![weather_tool()],
        output_schema: None,
        gemini_search_mode: None,
        tool_choice: GeminiToolChoice::Auto,
        thought_summary_display: GeminiThoughtSummaryDisplay::Hidden,
    };

    let request =
        build_generate_content_request(&prompt, &model_info, Some(ReasoningEffort::Low)).unwrap();
    let value = serde_json::to_value(request).unwrap();

    assert_eq!(
        value["contents"][2]["parts"],
        json!([
            {
                "functionResponse": {
                    "name": "get_weather",
                    "response": {
                        "output": "Image output attached.",
                        "success": true
                    },
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/webp",
                                "data": "CCC"
                            }
                        }
                    ]
                }
            },
            {
                "functionResponse": {
                    "name": "get_weather",
                    "response": {"output": "weather London"}
                }
            }
        ])
    );
}

#[test]
fn normalizes_system_instruction_without_truncating_content() {
    let mut catalog = gemini_model_catalog();
    let model_info = catalog.models.remove(0);
    let important_middle_rule = "IMPORTANT_MIDDLE_RULE_DO_NOT_DROP";
    let prompt = GeminiPrompt {
        instructions: format!("\n\nfirst rule\n\n\n{important_middle_rule}\n\n\nlast rule\n\n"),
        input: vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "Say ok.".to_string(),
            }],
            phase: None,
        }],
        tools: Vec::new(),
        output_schema: None,
        gemini_search_mode: None,
        tool_choice: GeminiToolChoice::Auto,
        thought_summary_display: GeminiThoughtSummaryDisplay::Hidden,
    };

    let request =
        build_generate_content_request(&prompt, &model_info, Some(ReasoningEffort::Low)).unwrap();
    let value = serde_json::to_value(request).unwrap();

    assert_eq!(
        value["systemInstruction"]["parts"][0]["text"],
        json!(format!(
            "first rule\n\n{important_middle_rule}\n\nlast rule"
        ))
    );
}

#[test]
fn serializes_native_response_schema_without_tools() {
    let mut catalog = gemini_model_catalog();
    let model_info = catalog.models.remove(0);
    let schema = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "outcome": {
                "type": "string",
                "enum": ["allow", "deny"]
            }
        },
        "required": ["outcome"]
    });
    let prompt = GeminiPrompt {
        instructions: "Return structured JSON.".to_string(),
        input: vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "Allow this.".to_string(),
            }],
            phase: None,
        }],
        tools: Vec::new(),
        output_schema: Some(schema),
        gemini_search_mode: None,
        tool_choice: GeminiToolChoice::Auto,
        thought_summary_display: GeminiThoughtSummaryDisplay::Hidden,
    };

    let request =
        build_generate_content_request(&prompt, &model_info, Some(ReasoningEffort::Low)).unwrap();
    let value = serde_json::to_value(request).unwrap();

    assert_eq!(
        value["generationConfig"]["responseMimeType"],
        json!("application/json")
    );
    assert_eq!(
        value["generationConfig"]["responseSchema"],
        json!({
            "type": "object",
            "properties": {
                "outcome": {
                    "type": "string",
                    "enum": ["allow", "deny"]
                }
            },
            "required": ["outcome"]
        })
    );
    assert!(value.get("tools").is_none());
    assert!(value.get("toolConfig").is_none());
}

#[test]
fn serializes_any_function_calling_config_for_plan_mode_fallback() {
    let mut catalog = gemini_model_catalog();
    let model_info = catalog.models.remove(0);
    let prompt = GeminiPrompt {
        instructions: String::new(),
        input: vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "Ask for clarification.".to_string(),
            }],
            phase: None,
        }],
        tools: vec![ToolSpec::Function(ResponsesApiTool {
            name: "request_user_input".to_string(),
            description: "Ask the user for clarification.".to_string(),
            strict: false,
            defer_loading: None,
            parameters: Default::default(),
            output_schema: None,
        })],
        output_schema: None,
        gemini_search_mode: None,
        tool_choice: GeminiToolChoice::Any {
            allowed_function_names: vec![
                "request_user_input".to_string(),
                "propose_plan".to_string(),
            ],
        },
        thought_summary_display: GeminiThoughtSummaryDisplay::Hidden,
    };

    let request =
        build_generate_content_request(&prompt, &model_info, Some(ReasoningEffort::Low)).unwrap();
    let value = serde_json::to_value(request).unwrap();

    assert_eq!(
        value["toolConfig"]["functionCallingConfig"],
        json!({
            "mode": "ANY",
            "allowedFunctionNames": ["request_user_input", "propose_plan"]
        })
    );
}

#[test]
fn structured_output_never_uses_forced_any_tool_config() {
    let mut catalog = gemini_model_catalog();
    let model_info = catalog.models.remove(0);
    let prompt = GeminiPrompt {
        instructions: String::new(),
        input: vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "Return JSON.".to_string(),
            }],
            phase: None,
        }],
        tools: vec![ToolSpec::Function(ResponsesApiTool {
            name: "request_user_input".to_string(),
            description: "Ask the user for clarification.".to_string(),
            strict: false,
            defer_loading: None,
            parameters: Default::default(),
            output_schema: None,
        })],
        output_schema: Some(json!({
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"}
            },
            "required": ["ok"]
        })),
        gemini_search_mode: None,
        tool_choice: GeminiToolChoice::Any {
            allowed_function_names: vec!["request_user_input".to_string()],
        },
        thought_summary_display: GeminiThoughtSummaryDisplay::Hidden,
    };

    let request =
        build_generate_content_request(&prompt, &model_info, Some(ReasoningEffort::Low)).unwrap();
    let value = serde_json::to_value(request).unwrap();

    assert_eq!(
        value["toolConfig"]["functionCallingConfig"],
        json!({"mode": "AUTO"})
    );
    assert_eq!(
        value["generationConfig"]["responseMimeType"],
        json!("application/json")
    );
}

fn weather_tool() -> ToolSpec {
    ToolSpec::Function(ResponsesApiTool {
        name: "get_weather".to_string(),
        description: "Get weather for a city.".to_string(),
        strict: false,
        defer_loading: None,
        parameters: JsonSchema::object(
            BTreeMap::from([(
                "city".to_string(),
                JsonSchema::string(Some("City name.".to_string())),
            )]),
            Some(vec!["city".to_string()]),
            None,
        ),
        output_schema: None,
    })
}

fn request_value_for_function_output(output: FunctionCallOutputPayload) -> serde_json::Value {
    let mut catalog = gemini_model_catalog();
    let model_info = catalog.models.remove(0);
    let prompt = GeminiPrompt {
        instructions: "You are Codex.".to_string(),
        input: vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "run the step tool".to_string(),
                }],
                phase: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "get_weather".to_string(),
                namespace: None,
                arguments: r#"{"city":"Paris"}"#.to_string(),
                call_id: "call-1".to_string(),
                thought_signature: Some("sig-1".to_string()),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call-1".to_string(),
                output,
            },
        ],
        tools: vec![weather_tool()],
        output_schema: None,
        gemini_search_mode: None,
        tool_choice: GeminiToolChoice::Auto,
        thought_summary_display: GeminiThoughtSummaryDisplay::Hidden,
    };

    let request =
        build_generate_content_request(&prompt, &model_info, Some(ReasoningEffort::High)).unwrap();
    serde_json::to_value(request).unwrap()
}

fn request_value_with_grounding(
    tools: Vec<ToolSpec>,
    output_schema: Option<Value>,
    google_search_grounding: GoogleSearchGrounding,
) -> serde_json::Value {
    let mut catalog = gemini_model_catalog();
    let model_info = catalog.models.remove(0);
    let prompt = GeminiPrompt {
        instructions: String::new(),
        input: vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "Use available tools if needed.".to_string(),
            }],
            phase: None,
        }],
        tools,
        output_schema,
        gemini_search_mode: None,
        tool_choice: GeminiToolChoice::Auto,
        thought_summary_display: GeminiThoughtSummaryDisplay::Hidden,
    };

    let request = build_generate_content_request_with_grounding(
        &prompt,
        &model_info,
        Some(ReasoningEffort::Low),
        google_search_grounding,
    )
    .unwrap();
    serde_json::to_value(request).unwrap()
}

#[test]
fn skips_web_search_call_when_translating_history() {
    let store = SignatureStore::default();
    let items = vec![
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "what's the weather?".to_string(),
            }],
            phase: None,
        },
        ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: "It is sunny.".to_string(),
            }],
            phase: None,
        },
        // Display-only grounding artifact persisted from a prior turn.
        ResponseItem::WebSearchCall {
            id: None,
            status: Some("completed".to_string()),
            action: Some(WebSearchAction::Search {
                query: None,
                queries: Some(vec!["weather".to_string()]),
            }),
        },
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "and tomorrow?".to_string(),
            }],
            phase: None,
        },
    ];

    let contents = contents_from_response_items(&items, &store)
        .expect("WebSearchCall must be skipped, not error");

    // The WebSearchCall contributes no Content; only the three messages survive, in order.
    assert_eq!(contents.len(), 3);
    let roles: Vec<&str> = contents
        .iter()
        .map(|content| content.role.as_str())
        .collect();
    assert_eq!(roles, vec!["user", "model", "user"]);
    assert_eq!(
        contents[0].parts[0].text.as_deref(),
        Some("what's the weather?")
    );
    assert_eq!(contents[1].parts[0].text.as_deref(), Some("It is sunny."));
    assert_eq!(contents[2].parts[0].text.as_deref(), Some("and tomorrow?"));
}
