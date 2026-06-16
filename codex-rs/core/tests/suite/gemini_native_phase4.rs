#![allow(clippy::expect_used)]

use std::collections::HashMap;
use std::time::Duration;

use super::gemini_native_phase2::gemini_builder as mock_gemini_builder;
use super::gemini_native_phase2::gemini_function_call_sse;
use super::gemini_native_phase2::gemini_text_sse;
use super::gemini_native_phase2::mount_gemini_sse_sequence;
use anyhow::Result;
use anyhow::bail;
use codex_features::Feature;
use codex_gemini_adapter::GEMINI_3_5_FLASH_MODEL;
use codex_gemini_adapter::GEMINI_API_KEY_ENV_VAR;
use codex_gemini_adapter::GOOGLE_CLOUD_LOCATION_ENV_VAR;
use codex_gemini_adapter::GOOGLE_CLOUD_PROJECT_ENV_VAR;
use codex_gemini_adapter::GOOGLE_GENAI_USE_VERTEXAI_ENV_VAR;
use codex_gemini_adapter::model_config::gemini_model_catalog;
use codex_model_provider_info::GEMINI_PROVIDER_ID;
use codex_model_provider_info::ModelProviderInfo;
use codex_protocol::config_types::CollaborationMode;
use codex_protocol::config_types::ModeKind;
use codex_protocol::config_types::Settings;
use codex_protocol::models::PermissionProfile;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::protocol::AskForApproval;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::request_user_input::RequestUserInputAnswer;
use codex_protocol::request_user_input::RequestUserInputResponse;
use codex_protocol::user_input::UserInput;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodex;
use core_test_support::test_codex::TestCodexBuilder;
use core_test_support::test_codex::TestCodexHarness;
use core_test_support::test_codex::test_codex;
use core_test_support::test_codex::turn_permission_fields;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use serial_test::serial;
use tokio::time::timeout;

const LIVE_GATE_TIMEOUT: Duration = Duration::from_secs(180);

fn live_gemini_builder() -> TestCodexBuilder {
    test_codex().with_config(|config| {
        let provider = ModelProviderInfo::create_gemini_provider();
        config.model = Some(GEMINI_3_5_FLASH_MODEL.to_string());
        config.model_catalog = Some(gemini_model_catalog());
        config.model_provider_id = GEMINI_PROVIDER_ID.to_string();
        config.model_provider = provider;
    })
}

fn function_declaration_names(request: &Value) -> Vec<String> {
    request["tools"][0]["functionDeclarations"]
        .as_array()
        .expect("function declarations")
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn live_gemini_auth_configured() -> bool {
    if std::env::var_os(GEMINI_API_KEY_ENV_VAR).is_some() {
        return true;
    }

    std::env::var_os(GOOGLE_GENAI_USE_VERTEXAI_ENV_VAR)
        .and_then(|value| value.into_string().ok())
        .is_some_and(|value| value.eq_ignore_ascii_case("true"))
        && std::env::var_os(GOOGLE_CLOUD_PROJECT_ENV_VAR).is_some()
        && std::env::var_os(GOOGLE_CLOUD_LOCATION_ENV_VAR).is_some()
}

async fn submit_turn(
    test: &TestCodex,
    text: &str,
    collaboration_mode: CollaborationMode,
) -> Result<String> {
    let (sandbox_policy, permission_profile) =
        turn_permission_fields(PermissionProfile::Disabled, test.config.cwd.as_path());
    test.codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: text.to_string(),
                text_elements: Vec::new(),
            }],
            environments: None,
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: codex_protocol::protocol::ThreadSettingsOverrides {
                cwd: Some(test.config.cwd.to_path_buf()),
                approval_policy: Some(AskForApproval::Never),
                sandbox_policy: Some(sandbox_policy),
                permission_profile,
                collaboration_mode: Some(collaboration_mode),
                ..Default::default()
            },
        })
        .await?;

    loop {
        let event = timeout(LIVE_GATE_TIMEOUT, test.codex.next_event())
            .await
            .expect("timeout waiting for turn start")
            .expect("stream ended unexpectedly while waiting for turn start");
        if let EventMsg::TurnStarted(started) = event.msg {
            return Ok(started.turn_id);
        }
    }
}

async fn collect_events_until_turn_complete(
    test: &TestCodex,
    turn_id: &str,
) -> Result<Vec<EventMsg>> {
    let mut events = Vec::new();
    loop {
        let event = timeout(LIVE_GATE_TIMEOUT, test.codex.next_event())
            .await
            .expect("timeout waiting for turn event")
            .expect("stream ended unexpectedly while waiting for turn event");
        match &event.msg {
            EventMsg::TurnComplete(complete) if complete.turn_id == turn_id => {
                events.push(event.msg);
                return Ok(events);
            }
            EventMsg::Error(error) => bail!("turn failed: {}", error.message),
            _ => events.push(event.msg),
        }
    }
}

fn default_collaboration_mode(model: String) -> CollaborationMode {
    CollaborationMode {
        mode: ModeKind::Default,
        settings: Settings {
            model,
            reasoning_effort: Some(ReasoningEffort::Low),
            developer_instructions: None,
        },
    }
}

fn plan_collaboration_mode(model: String) -> CollaborationMode {
    CollaborationMode {
        mode: ModeKind::Plan,
        settings: Settings {
            model,
            reasoning_effort: Some(ReasoningEffort::Low),
            developer_instructions: None,
        },
    }
}

fn final_agent_message(events: &[EventMsg], turn_id: &str) -> String {
    events
        .iter()
        .rev()
        .find_map(|event| match event {
            EventMsg::TurnComplete(complete) if complete.turn_id == turn_id => {
                complete.last_agent_message.clone()
            }
            _ => None,
        })
        .or_else(|| {
            events.iter().rev().find_map(|event| match event {
                EventMsg::AgentMessage(message) => Some(message.message.clone()),
                _ => None,
            })
        })
        .unwrap_or_default()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_code_mode_falls_back_to_direct_json_tools() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let harness = TestCodexHarness::with_builder(mock_gemini_builder().with_config(|config| {
        config
            .features
            .enable(Feature::CodeMode)
            .expect("test config should allow feature update");
    }))
    .await?;
    let requests = mount_gemini_sse_sequence(
        harness.server(),
        vec![
            gemini_function_call_sse(
                "exec_command",
                json!({ "cmd": "printf phase4-code-mode" }),
                Some("sig-code-mode"),
            ),
            gemini_text_sse("done"),
        ],
    )
    .await;

    harness
        .test()
        .submit_turn_with_permission_profile(
            "Use code mode to print the phase4 marker.",
            PermissionProfile::Disabled,
        )
        .await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), 2);
    let tool_names = function_declaration_names(&captured[0]);
    assert!(
        tool_names.contains(&"exec_command".to_string()),
        "Gemini code-mode fallback must expose direct exec_command: {tool_names:?}"
    );
    assert!(
        !tool_names.contains(&codex_code_mode::PUBLIC_TOOL_NAME.to_string())
            && !tool_names.contains(&codex_code_mode::WAIT_TOOL_NAME.to_string()),
        "Gemini code-mode fallback must not expose code-mode exec/wait tools: {tool_names:?}"
    );

    let function_response = captured[1]["contents"]
        .as_array()
        .expect("follow-up contents")
        .iter()
        .flat_map(|content| content["parts"].as_array().expect("content parts").iter())
        .filter_map(|part| part.get("functionResponse"))
        .find(|response| response["name"].as_str() == Some("exec_command"))
        .expect("exec_command functionResponse");
    let function_response_json =
        serde_json::to_string(&function_response["response"]).expect("response JSON");
    assert!(
        function_response_json.contains("phase4-code-mode"),
        "direct exec_command functionResponse must include output: {function_response_json}"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_subagent_tools_are_flat_multi_agent_v2_functions() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let harness = TestCodexHarness::with_builder(mock_gemini_builder()).await?;
    let requests = mount_gemini_sse_sequence(harness.server(), vec![gemini_text_sse("done")]).await;

    harness
        .test()
        .submit_turn_with_permission_profile(
            "List the available agent tools.",
            PermissionProfile::Disabled,
        )
        .await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), 1);
    assert!(
        captured[0]["systemInstruction"]["parts"][0]["text"]
            .as_str()
            .is_some_and(|instructions| {
                instructions.contains("Gemini subagent coordination")
                    && instructions.contains("call `wait_agent` again")
            }),
        "Gemini subagent requests must include coordination instructions: {:?}",
        captured[0]["systemInstruction"]
    );
    let tool_names = function_declaration_names(&captured[0]);
    assert!(
        tool_names.contains(&"spawn_agent".to_string()),
        "Gemini request must expose flat spawn_agent: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"wait_agent".to_string()),
        "Gemini request must expose flat wait_agent: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"send_message".to_string()),
        "Gemini request must expose flat send_message: {tool_names:?}"
    );
    assert!(
        !tool_names.contains(&"multi_agent_v1".to_string()),
        "Gemini request must not expose the V1 namespace tool: {tool_names:?}"
    );
    let spawn_agent_declaration = captured[0]["tools"][0]["functionDeclarations"]
        .as_array()
        .expect("function declarations")
        .iter()
        .find(|tool| tool["name"].as_str() == Some("spawn_agent"))
        .expect("spawn_agent function declaration");
    assert!(
        spawn_agent_declaration["description"]
            .as_str()
            .is_some_and(|description| description
                .contains("wait_agent` until you have received a final-status notification")
                && description.contains(
                    "deliver your complete result as your final-channel plain-text message"
                )),
        "Gemini spawn_agent declaration must include subagent wait guidance: {spawn_agent_declaration:?}"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn live_gemini_plan_mode_requests_user_input_before_mutation() -> Result<()> {
    skip_if_no_network!(Ok(()));
    if !live_gemini_auth_configured() {
        eprintln!("skipping live Gemini Phase 4 plan-mode gate: no Gemini auth configured");
        return Ok(());
    }

    let harness = TestCodexHarness::with_builder(live_gemini_builder()).await?;
    let turn_id = submit_turn(
        harness.test(),
        "Improve this repository, but the scope is intentionally ambiguous. In Plan mode, ask me what specific area I want changed before you inspect, edit, or run tools.",
        plan_collaboration_mode(GEMINI_3_5_FLASH_MODEL.to_string()),
    )
    .await?;

    loop {
        let event = timeout(LIVE_GATE_TIMEOUT, harness.test().codex.next_event())
            .await
            .expect("timeout waiting for plan-mode event")
            .expect("stream ended unexpectedly while waiting for plan-mode event");
        match event.msg {
            EventMsg::RequestUserInput(request) if request.turn_id == turn_id => {
                assert!(!request.call_id.is_empty());
                assert!(!request.questions.is_empty());
                let answers = request
                    .questions
                    .iter()
                    .map(|question| {
                        (
                            question.id.clone(),
                            RequestUserInputAnswer {
                                answers: vec![
                                    "Do not edit. Reply with a short plan for inspecting the code-mode harness."
                                        .to_string(),
                                ],
                            },
                        )
                    })
                    .collect::<HashMap<_, _>>();
                harness
                    .test()
                    .codex
                    .submit(Op::UserInputAnswer {
                        id: request.turn_id,
                        response: RequestUserInputResponse { answers },
                    })
                    .await?;
                let _events = collect_events_until_turn_complete(harness.test(), &turn_id).await?;
                return Ok(());
            }
            EventMsg::TurnComplete(complete) if complete.turn_id == turn_id => {
                bail!("Gemini completed the Plan-mode turn before requesting user input");
            }
            EventMsg::ExecCommandBegin(exec) if exec.turn_id == turn_id => {
                bail!(
                    "Gemini started a command before requesting Plan-mode input: {:?}",
                    exec.command
                );
            }
            EventMsg::PatchApplyBegin(_)
            | EventMsg::PatchApplyUpdated(_)
            | EventMsg::PatchApplyEnd(_) => {
                bail!("Gemini applied or began a patch before requesting Plan-mode input");
            }
            EventMsg::Error(error) => bail!("Gemini Plan-mode turn failed: {}", error.message),
            _ => {}
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn live_gemini_code_mode_uses_direct_json_tool_fallback() -> Result<()> {
    skip_if_no_network!(Ok(()));
    if !live_gemini_auth_configured() {
        eprintln!("skipping live Gemini Phase 4 code-mode gate: no Gemini auth configured");
        return Ok(());
    }

    let harness = TestCodexHarness::with_builder(live_gemini_builder().with_config(|config| {
        config
            .features
            .enable(Feature::CodeMode)
            .expect("test config should allow feature update");
    }))
    .await?;
    let prompt = r#"CodeMode is enabled, but for Gemini use direct JSON function tools. Call `exec_command` with command `printf gemini-code-mode-ok`, then reply with the output."#;
    let turn_id = submit_turn(
        harness.test(),
        prompt,
        default_collaboration_mode(GEMINI_3_5_FLASH_MODEL.to_string()),
    )
    .await?;
    let events = collect_events_until_turn_complete(harness.test(), &turn_id).await?;

    let exec_command_arguments = events.iter().find_map(|event| match event {
        EventMsg::RawResponseItem(raw) => match &raw.item {
            ResponseItem::FunctionCall {
                name, arguments, ..
            } if name == "exec_command" => Some(arguments.clone()),
            _ => None,
        },
        _ => None,
    });
    let exec_command_arguments = exec_command_arguments
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("Gemini did not call the direct exec_command function"))?;
    assert!(
        exec_command_arguments.contains("gemini-code-mode-ok"),
        "Gemini direct exec_command call must include the marker command: {exec_command_arguments}"
    );
    assert!(
        events.iter().all(|event| {
            !matches!(
                event,
                EventMsg::RawResponseItem(raw)
                    if matches!(
                        &raw.item,
                        ResponseItem::FunctionCall { name, .. }
                            if name == codex_code_mode::PUBLIC_TOOL_NAME
                    )
            )
        }),
        "Gemini code-mode fallback must not use code-mode exec: {events:?}"
    );
    assert!(
        events.iter().any(|event| {
            matches!(
                event,
                EventMsg::ExecCommandBegin(exec)
                    if exec.turn_id == turn_id
                        && exec.command.iter().any(|arg| arg.contains("gemini-code-mode-ok"))
            )
        }),
        "Gemini code-mode fallback must dispatch direct exec_command: {events:?}"
    );
    assert!(
        final_agent_message(&events, &turn_id).contains("gemini-code-mode-ok"),
        "Gemini final response must include nested command output: {events:?}"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn live_gemini_subagents_spawn_wait_and_complete() -> Result<()> {
    skip_if_no_network!(Ok(()));
    if !live_gemini_auth_configured() {
        eprintln!("skipping live Gemini Phase 4 subagent gate: no Gemini auth configured");
        return Ok(());
    }

    let harness = TestCodexHarness::with_builder(live_gemini_builder()).await?;
    let prompt = r#"Use the subagent tools. Spawn exactly two subagents, one task_name `alpha` with message `Reply only ALPHA_RESULT=4`, and one task_name `beta` with message `Reply only BETA_RESULT=8`. Wait for both to complete with wait_agent. Then reply exactly `ALPHA_RESULT=4; BETA_RESULT=8`."#;
    let turn_id = submit_turn(
        harness.test(),
        prompt,
        default_collaboration_mode(GEMINI_3_5_FLASH_MODEL.to_string()),
    )
    .await?;
    let events = collect_events_until_turn_complete(harness.test(), &turn_id).await?;

    let spawn_events = events
        .iter()
        .filter_map(|event| match event {
            EventMsg::CollabAgentSpawnEnd(spawn) => Some(spawn),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        spawn_events.len() >= 2,
        "Gemini parent must spawn at least two subagents: {events:?}"
    );
    assert!(
        spawn_events.iter().all(|spawn| {
            spawn.new_thread_id.is_some() && spawn.model == GEMINI_3_5_FLASH_MODEL
        }),
        "spawned agents must inherit the Gemini Flash model: {spawn_events:?}"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, EventMsg::CollabWaitingEnd(_))),
        "Gemini parent must call wait_agent: {events:?}"
    );
    let final_message = final_agent_message(&events, &turn_id);
    assert!(
        final_message.contains("ALPHA_RESULT=4") && final_message.contains("BETA_RESULT=8"),
        "Gemini parent final response must include both subagent results: {final_message}"
    );

    Ok(())
}
