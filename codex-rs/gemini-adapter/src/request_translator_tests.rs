use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use pretty_assertions::assert_eq;
use serde_json::json;

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
