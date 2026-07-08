use anyhow::Result;
use codex_collaboration_mode_templates::PLAN;
use codex_collaboration_mode_templates::plan_for_gemini;
use codex_protocol::config_types::CollaborationMode;
use codex_protocol::config_types::ModeKind;
use codex_protocol::config_types::Settings;
use codex_protocol::protocol::COLLABORATION_MODE_CLOSE_TAG;
use codex_protocol::protocol::COLLABORATION_MODE_OPEN_TAG;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::protocol::ThreadSettingsOverrides;
use codex_protocol::user_input::UserInput;
use core_test_support::skip_if_no_network;
use core_test_support::submit_thread_settings;
use core_test_support::test_codex::TestCodexHarness;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::Value;

use super::gemini_native_phase2::gemini_builder;
use super::gemini_native_phase2::gemini_text_sse;
use super::gemini_native_phase2::mount_gemini_sse_sequence;

const PLAN_QUALITY_CONTRACT_MARKER: &str = "## Plan quality contract (strict)";

fn gemini_text_parts(request: &Value) -> Vec<&str> {
    request["contents"]
        .as_array()
        .expect("Gemini contents")
        .iter()
        .flat_map(|content| content["parts"].as_array().expect("content parts"))
        .filter_map(|part| part["text"].as_str())
        .collect()
}

fn gemini_system_instruction(request: &Value) -> &str {
    request["systemInstruction"]["parts"][0]["text"]
        .as_str()
        .expect("Gemini request systemInstruction text")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_plan_mode_stock_template_reaches_contents_with_quality_contract() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let harness = TestCodexHarness::with_builder(gemini_builder()).await?;
    let model = harness.test().session_configured.model.clone();
    submit_thread_settings(
        harness.test().codex.as_ref(),
        ThreadSettingsOverrides {
            collaboration_mode: Some(CollaborationMode {
                mode: ModeKind::Plan,
                settings: Settings {
                    model,
                    reasoning_effort: None,
                    developer_instructions: Some(PLAN.to_string()),
                },
            }),
            ..Default::default()
        },
    )
    .await?;

    let requests = mount_gemini_sse_sequence(harness.server(), vec![gemini_text_sse("done")]).await;
    harness
        .test()
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "draft a plan".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_event(&harness.test().codex, |ev| {
        matches!(ev, EventMsg::TurnComplete(_))
    })
    .await;

    let captured = requests.requests();
    assert_eq!(captured.len(), 1);

    let expected_collaboration_mode = format!(
        "{COLLABORATION_MODE_OPEN_TAG}{}{COLLABORATION_MODE_CLOSE_TAG}",
        plan_for_gemini()
    );
    let text_parts = gemini_text_parts(&captured[0]);
    assert_eq!(
        text_parts
            .iter()
            .filter(|text| text.contains(&expected_collaboration_mode))
            .count(),
        1
    );
    assert!(
        !gemini_system_instruction(&captured[0]).contains(PLAN_QUALITY_CONTRACT_MARKER),
        "Gemini systemInstruction must not include plan quality marker"
    );

    Ok(())
}
