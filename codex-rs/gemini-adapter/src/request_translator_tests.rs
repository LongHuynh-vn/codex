use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;
use codex_tools::AdditionalProperties;
use codex_tools::JsonSchema;
use codex_tools::JsonSchemaPrimitiveType;
use codex_tools::JsonSchemaType;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::collections::BTreeMap;

use super::*;
use crate::GeminiPrompt;
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
