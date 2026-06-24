use super::*;
use crate::StateDbHandle;
use crate::ThreadManager;
use crate::config::AgentRoleConfig;
use crate::config::DEFAULT_AGENT_MAX_DEPTH;
use crate::context::SubagentNotification;
use crate::function_tool::FunctionCallError;
use crate::goals::GoalRuntimeEvent;
use crate::init_state_db;
use crate::session::TurnInput;
use crate::session::tests::make_session_and_context;
use crate::session::tests::make_session_and_context_with_rx;
use crate::session_prefix::format_subagent_notification_message;
use crate::state::TaskKind;
use crate::tasks::SessionTask;
use crate::tasks::SessionTaskContext;
use crate::thread_manager::thread_store_from_config;
use crate::tools::context::ToolOutput;
use crate::tools::handlers::UpdateGoalHandler;
use crate::tools::handlers::multi_agents_spec::WaitAgentTimeoutOptions;
use crate::tools::handlers::multi_agents_spec::WaitAgentV2OutputMode;
use crate::tools::handlers::multi_agents_v2::CloseAgentHandler as CloseAgentHandlerV2;
use crate::tools::handlers::multi_agents_v2::FollowupTaskHandler as FollowupTaskHandlerV2;
use crate::tools::handlers::multi_agents_v2::ListAgentsHandler as ListAgentsHandlerV2;
use crate::tools::handlers::multi_agents_v2::SendMessageHandler as SendMessageHandlerV2;
use crate::tools::handlers::multi_agents_v2::SpawnAgentHandler as SpawnAgentHandlerV2;
use crate::tools::handlers::multi_agents_v2::WaitAgentHandler as WaitAgentHandlerV2;
use crate::tools::handlers::multi_agents_v2::wait::GEMINI_MULTI_AGENT_V2_WAIT_TIMEOUT_FLOOR_MS;
use crate::tools::handlers::multi_agents_v2::wait::effective_wait_agent_v2_timeout_options;
use crate::tools::handlers::multi_agents_v2::wait::floor_gemini_wait_timeout_ms;
use crate::turn_diff_tracker::TurnDiffTracker;
use codex_extension_api::empty_extension_registry;
use codex_features::Feature;
use codex_login::AuthManager;
use codex_login::CodexAuth;
use codex_model_provider::create_model_provider;
use codex_model_provider_info::AMAZON_BEDROCK_PROVIDER_ID;
use codex_model_provider_info::GEMINI_PROVIDER_ID;
use codex_model_provider_info::ModelProviderInfo;
use codex_model_provider_info::WireApi;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::AgentPath;
use codex_protocol::ThreadId;
use codex_protocol::config_types::ServiceTier;
use codex_protocol::config_types::ShellEnvironmentPolicy;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::MessagePhase;
use codex_protocol::models::PermissionProfile;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::models::SandboxEnforcement;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::protocol::AgentStatus;
use codex_protocol::protocol::AskForApproval;
use codex_protocol::protocol::CollabAgentRef;
use codex_protocol::protocol::CollabAgentStatusEntry;
use codex_protocol::protocol::CollabWaitingEndEvent;
use codex_protocol::protocol::ErrorEvent;
use codex_protocol::protocol::Event;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::FileSystemAccessMode;
use codex_protocol::protocol::FileSystemPath;
use codex_protocol::protocol::FileSystemSandboxEntry;
use codex_protocol::protocol::FileSystemSandboxPolicy;
use codex_protocol::protocol::InitialHistory;
use codex_protocol::protocol::InterAgentCommunication;
use codex_protocol::protocol::InternalSessionSource;
use codex_protocol::protocol::NetworkSandboxPolicy;
use codex_protocol::protocol::Op;
use codex_protocol::protocol::RolloutItem;
use codex_protocol::protocol::SandboxPolicy;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::SubAgentSource;
use codex_protocol::protocol::TurnAbortReason;
use codex_protocol::protocol::TurnAbortedEvent;
use codex_protocol::protocol::TurnCompleteEvent;
use codex_protocol::user_input::UserInput;
use codex_state::DirectionalThreadSpawnEdgeStatus;
use core_test_support::TempDirExt;
use pretty_assertions::assert_eq;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::sync::Notify;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

fn invocation(
    session: Arc<crate::session::session::Session>,
    turn: Arc<TurnContext>,
    tool_name: &str,
    payload: ToolPayload,
) -> ToolInvocation {
    ToolInvocation {
        session,
        turn,
        cancellation_token: CancellationToken::new(),
        tracker: Arc::new(Mutex::new(TurnDiffTracker::default())),
        call_id: "call-1".to_string(),
        tool_name: codex_tools::ToolName::plain(tool_name),
        source: crate::tools::context::ToolCallSource::Direct,
        payload,
    }
}

fn function_payload(args: serde_json::Value) -> ToolPayload {
    ToolPayload::Function {
        arguments: args.to_string(),
    }
}

fn parse_agent_id(id: &str) -> ThreadId {
    ThreadId::from_string(id).expect("agent id should be valid")
}

fn thread_manager() -> ThreadManager {
    ThreadManager::with_models_provider_for_tests(
        CodexAuth::from_api_key("dummy"),
        built_in_model_providers(/* openai_base_url */ /*openai_base_url*/ None)["openai"].clone(),
    )
}

async fn install_role_with_model_override(turn: &mut TurnContext) -> String {
    let role_name = "fork-context-role".to_string();
    tokio::fs::create_dir_all(&turn.config.codex_home)
        .await
        .expect("codex home should be created");
    let role_config_path = turn
        .config
        .codex_home
        .as_path()
        .join("fork-context-role.toml");
    tokio::fs::write(
        &role_config_path,
        r#"model = "gpt-5-role-override"
model_provider = "ollama"
model_reasoning_effort = "minimal"
"#,
    )
    .await
    .expect("role config should be written");

    let mut config = (*turn.config).clone();
    config.agent_roles.insert(
        role_name.clone(),
        AgentRoleConfig {
            description: Some("Role with model overrides".to_string()),
            config_file: Some(role_config_path),
            nickname_candidates: None,
        },
    );
    turn.config = Arc::new(config);

    role_name
}

fn set_turn_config(turn: &mut TurnContext, config: crate::config::Config) {
    turn.multi_agent_version = config.multi_agent_version_from_features();
    turn.config = Arc::new(config);
}

fn use_bedrock_provider(turn: &mut TurnContext) {
    let provider_info = ModelProviderInfo::create_amazon_bedrock_provider(/*aws*/ None);
    let mut config = (*turn.config).clone();
    config.model_provider_id = AMAZON_BEDROCK_PROVIDER_ID.to_string();
    config.model_provider = provider_info.clone();
    turn.provider = create_model_provider(provider_info, turn.auth_manager.clone());
    turn.config = Arc::new(config);
}

fn use_gemini_provider(turn: &mut TurnContext) {
    let mut provider_info = ModelProviderInfo::create_gemini_provider();
    provider_info.experimental_bearer_token = Some("test-gemini-key".to_string());
    let mut config = (*turn.config).clone();
    config.model_provider_id = GEMINI_PROVIDER_ID.to_string();
    config.model_provider = provider_info.clone();
    turn.provider = create_model_provider(provider_info, turn.auth_manager.clone());
    turn.config = Arc::new(config);
}

enum AutoGoalProvider {
    Gemini,
    Bedrock,
}

struct AutoGoalSpawnSetup {
    manager: ThreadManager,
    session: Arc<crate::session::session::Session>,
    turn: Arc<TurnContext>,
    state_db: StateDbHandle,
}

async fn auto_goal_spawn_setup<F>(
    provider: AutoGoalProvider,
    session_source: F,
    first_user_message: Option<&str>,
) -> AutoGoalSpawnSetup
where
    F: FnOnce(ThreadId) -> SessionSource,
{
    let (_session, mut turn) = make_session_and_context().await;
    let mut config = (*turn.config).clone();
    for feature in [Feature::MultiAgentV2, Feature::Goals, Feature::Sqlite] {
        config
            .features
            .enable(feature)
            .expect("test config should allow feature update");
    }
    set_turn_config(&mut turn, config);
    match provider {
        AutoGoalProvider::Gemini => use_gemini_provider(&mut turn),
        AutoGoalProvider::Bedrock => use_bedrock_provider(&mut turn),
    }
    let config = (*turn.config).clone();
    let state_db = init_state_db(&config)
        .await
        .expect("sqlite state db should initialize");
    let manager = ThreadManager::with_models_provider_home_and_state_for_tests(
        CodexAuth::from_api_key("dummy"),
        config.model_provider.clone(),
        config.codex_home.to_path_buf(),
        Arc::new(codex_exec_server::EnvironmentManager::default_for_tests()),
        Some(state_db.clone()),
    );
    let root = manager
        .start_thread(config)
        .await
        .expect("root thread should start");
    if let Some(first_user_message) = first_user_message {
        root.thread
            .inject_user_message_without_turn(first_user_message.to_string())
            .await;
        root.thread
            .codex
            .session
            .flush_rollout()
            .await
            .expect("rollout should flush");
    }

    let session = root.thread.codex.session.clone();
    let mut turn = session.new_default_turn().await;
    Arc::get_mut(&mut turn)
        .expect("fresh turn should be unique")
        .session_source = session_source(root.thread_id);

    AutoGoalSpawnSetup {
        manager,
        session,
        turn,
        state_db,
    }
}

async fn spawn_agent_v2_for_auto_goal(setup: &AutoGoalSpawnSetup, task_name: &str) {
    SpawnAgentHandlerV2::default()
        .handle(invocation(
            setup.session.clone(),
            setup.turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": format!("inspect this repo for {task_name}"),
                "task_name": task_name,
                "fork_turns": "none"
            })),
        ))
        .await
        .expect("spawn_agent should succeed");
}

fn thread_spawn_source_for_test(
    parent_thread_id: ThreadId,
    agent_path: Option<AgentPath>,
) -> SessionSource {
    SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
        parent_thread_id,
        depth: 1,
        agent_path,
        agent_nickname: None,
        agent_role: None,
    })
}

#[test]
fn multi_agent_v2_wait_agent_effective_default_is_longer_for_gemini() {
    let config = crate::config::MultiAgentV2Config::default();

    let options = effective_wait_agent_v2_timeout_options(&config, WireApi::GeminiNative);

    assert_eq!(options.default_timeout_ms, 120_000);
}

#[test]
fn multi_agent_v2_wait_agent_effective_default_stays_configured_for_non_gemini() {
    let config = crate::config::MultiAgentV2Config::default();

    let options = effective_wait_agent_v2_timeout_options(&config, WireApi::Responses);

    assert_eq!(options.default_timeout_ms, 30_000);
}

#[test]
fn multi_agent_v2_wait_agent_gemini_effective_default_clamps_to_configured_max() {
    let mut config = crate::config::MultiAgentV2Config::default();
    config.max_wait_timeout_ms = 60_000;

    let options = effective_wait_agent_v2_timeout_options(&config, WireApi::GeminiNative);

    assert_eq!(options.default_timeout_ms, 60_000);
}

#[test]
fn multi_agent_v2_wait_agent_gemini_effective_default_clamps_to_configured_min() {
    let mut config = crate::config::MultiAgentV2Config::default();
    config.min_wait_timeout_ms = 180_000;
    config.max_wait_timeout_ms = 240_000;

    let options = effective_wait_agent_v2_timeout_options(&config, WireApi::GeminiNative);

    assert_eq!(options.default_timeout_ms, 180_000);
}

#[test]
fn multi_agent_v2_wait_agent_gemini_floor_timeout_ms_handles_short_and_capped_values() {
    let floor = GEMINI_MULTI_AGENT_V2_WAIT_TIMEOUT_FLOOR_MS;

    assert_eq!(
        floor_gemini_wait_timeout_ms(15_000, floor, 3_600_000),
        floor
    );
    assert_eq!(floor_gemini_wait_timeout_ms(floor, floor, 3_600_000), floor);
    assert_eq!(
        floor_gemini_wait_timeout_ms(180_000, floor, 3_600_000),
        180_000
    );
    assert_eq!(floor_gemini_wait_timeout_ms(15_000, floor, 60_000), 60_000);
}

async fn last_waiting_end_event(rx: &async_channel::Receiver<Event>) -> CollabWaitingEndEvent {
    let deadline = Duration::from_secs(2);
    let start = std::time::Instant::now();
    loop {
        let remaining = deadline.saturating_sub(start.elapsed());
        let event = timeout(remaining, rx.recv())
            .await
            .expect("timed out waiting for CollabWaitingEnd")
            .expect("event channel should be open");
        if let EventMsg::CollabWaitingEnd(event) = event.msg {
            return event;
        }
    }
}

fn expect_text_output<T>(output: T) -> (String, Option<bool>)
where
    T: ToolOutput,
{
    let response = output.to_response_item(
        "call-1",
        &ToolPayload::Function {
            arguments: "{}".to_string(),
        },
    );
    match response {
        ResponseInputItem::FunctionCallOutput { output, .. }
        | ResponseInputItem::CustomToolCallOutput { output, .. } => {
            let content = match output.body {
                FunctionCallOutputBody::Text(text) => text,
                FunctionCallOutputBody::ContentItems(items) => {
                    codex_protocol::models::function_call_output_content_items_to_text(&items)
                        .unwrap_or_default()
                }
            };
            (content, output.success)
        }
        other => panic!("expected function output, got {other:?}"),
    }
}

#[derive(Debug, Deserialize)]
struct ListAgentsResult {
    agents: Vec<ListedAgentResult>,
}

#[derive(Debug, Deserialize)]
struct ListedAgentResult {
    agent_name: String,
    agent_status: serde_json::Value,
    last_task_message: Option<String>,
}

#[tokio::test]
async fn handler_rejects_non_function_payloads() {
    let (session, turn) = make_session_and_context().await;
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "spawn_agent",
        ToolPayload::Custom {
            input: "hello".to_string(),
        },
    );
    let Err(err) = SpawnAgentHandler::default().handle(invocation).await else {
        panic!("payload should be rejected");
    };
    assert_eq!(
        err,
        FunctionCallError::RespondToModel(
            "collab handler received unsupported payload".to_string()
        )
    );
}

#[tokio::test]
async fn spawn_agent_rejects_empty_message() {
    let (session, turn) = make_session_and_context().await;
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "spawn_agent",
        function_payload(json!({"message": "   "})),
    );
    let Err(err) = SpawnAgentHandler::default().handle(invocation).await else {
        panic!("empty message should be rejected");
    };
    assert_eq!(
        err,
        FunctionCallError::RespondToModel("Empty message can't be sent to an agent".to_string())
    );
}

#[tokio::test]
async fn spawn_agent_rejects_when_message_and_items_are_both_set() {
    let (session, turn) = make_session_and_context().await;
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "spawn_agent",
        function_payload(json!({
            "message": "hello",
            "items": [{"type": "mention", "name": "drive", "path": "app://drive"}]
        })),
    );
    let Err(err) = SpawnAgentHandler::default().handle(invocation).await else {
        panic!("message+items should be rejected");
    };
    assert_eq!(
        err,
        FunctionCallError::RespondToModel(
            "Provide either message or items, but not both".to_string()
        )
    );
}

#[tokio::test]
async fn spawn_agent_uses_explorer_role_and_preserves_approval_policy() {
    #[derive(Debug, Deserialize)]
    struct SpawnAgentResult {
        agent_id: String,
        nickname: Option<String>,
    }

    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();
    let mut config = (*turn.config).clone();
    let provider_info =
        built_in_model_providers(/* openai_base_url */ /*openai_base_url*/ None)["ollama"].clone();
    config.model_provider_id = "ollama".to_string();
    config.model_provider = provider_info.clone();
    config
        .permissions
        .approval_policy
        .set(AskForApproval::OnRequest)
        .expect("approval policy should be set");
    turn.approval_policy
        .set(AskForApproval::OnRequest)
        .expect("approval policy should be set");
    turn.provider = create_model_provider(provider_info, turn.auth_manager.clone());
    turn.config = Arc::new(config);

    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "spawn_agent",
        function_payload(json!({
            "message": "inspect this repo",
            "agent_type": "explorer"
        })),
    );
    let output = SpawnAgentHandler::default()
        .handle(invocation)
        .await
        .expect("spawn_agent should succeed");
    let (content, _) = expect_text_output(output);
    let result: SpawnAgentResult =
        serde_json::from_str(&content).expect("spawn_agent result should be json");
    let agent_id = parse_agent_id(&result.agent_id);
    assert!(
        result
            .nickname
            .as_deref()
            .is_some_and(|nickname| !nickname.is_empty())
    );
    let snapshot = manager
        .get_thread(agent_id)
        .await
        .expect("spawned agent thread should exist")
        .config_snapshot()
        .await;
    assert_eq!(snapshot.approval_policy, AskForApproval::OnRequest);
    assert_eq!(snapshot.model_provider_id, "ollama");
}

#[tokio::test]
async fn spawn_agent_fork_context_rejects_agent_type_override() {
    let (mut session, mut turn) = make_session_and_context().await;
    let role_name = install_role_with_model_override(&mut turn).await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let err = SpawnAgentHandler::default()
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "agent_type": role_name,
                "fork_context": true
            })),
        ))
        .await
        .err()
        .expect("fork_context should reject agent_type overrides");

    assert_eq!(
        err,
        FunctionCallError::RespondToModel(
            "Full-history forked agents inherit the parent agent type, model, and reasoning effort; omit agent_type, model, and reasoning_effort, or spawn without a full-history fork.".to_string(),
        )
    );
}

#[tokio::test]
async fn spawn_agent_fork_context_rejects_child_model_overrides() {
    let (mut session, turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;

    let err = SpawnAgentHandler::default()
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "model": "gpt-5-child-override",
                "reasoning_effort": "low",
                "fork_context": true
            })),
        ))
        .await
        .err()
        .expect("forked spawn should reject child model overrides");

    assert_eq!(
        err,
            FunctionCallError::RespondToModel(
            "Full-history forked agents inherit the parent agent type, model, and reasoning effort; omit agent_type, model, and reasoning_effort, or spawn without a full-history fork.".to_string(),
        )
    );
}

#[tokio::test]
async fn multi_agent_v2_spawn_fork_turns_all_rejects_agent_type_override() {
    let (mut session, mut turn) = make_session_and_context().await;
    let role_name = install_role_with_model_override(&mut turn).await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    let turn = TurnContext {
        config: Arc::new(config),
        multi_agent_version: codex_protocol::protocol::MultiAgentVersion::V2,
        ..turn
    };

    let err = SpawnAgentHandlerV2::default()
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "task_name": "fork_context_v2",
                "agent_type": role_name,
                "fork_turns": "all"
            })),
        ))
        .await
        .err()
        .expect("fork_turns=all should reject agent_type overrides");

    assert_eq!(
        err,
        FunctionCallError::RespondToModel(
            "Full-history forked agents inherit the parent agent type, model, and reasoning effort; omit agent_type, model, and reasoning_effort, or spawn without a full-history fork.".to_string(),
        )
    );
}

#[tokio::test]
async fn multi_agent_v2_spawn_non_gemini_defaults_to_full_fork_and_rejects_child_model_overrides() {
    let (mut session, mut turn) = make_session_and_context().await;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);
    use_bedrock_provider(&mut turn);
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;

    let err = SpawnAgentHandlerV2::default()
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "task_name": "fork_context_v2",
                "model": "gpt-5-child-override",
                "reasoning_effort": "low"
            })),
        ))
        .await
        .err()
        .expect("default full fork should reject child model overrides");

    assert_eq!(
        err,
            FunctionCallError::RespondToModel(
            "Full-history forked agents inherit the parent agent type, model, and reasoning effort; omit agent_type, model, and reasoning_effort, or spawn without a full-history fork.".to_string(),
        )
    );
}

#[tokio::test]
async fn multi_agent_v2_spawn_gemini_default_omitted_or_empty_fork_turns_is_scoped() {
    #[derive(Debug, Deserialize)]
    struct SpawnAgentResult {
        task_name: String,
    }

    let parent_marker = "parent history marker must not reach Gemini default-scoped children";
    let (mut session, mut turn) = make_session_and_context().await;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);
    use_gemini_provider(&mut turn);
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    root.thread
        .inject_user_message_without_turn(parent_marker.to_string())
        .await;
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    for (task_name, fork_turns) in [
        ("gemini_default_omitted", None),
        ("gemini_default_empty", Some("   ")),
    ] {
        let child_message = format!("inspect this repo for {task_name}");
        let mut args = json!({
            "message": child_message.clone(),
            "task_name": task_name,
            "model": "gpt-5.4",
            "reasoning_effort": "low",
        });
        if let Some(fork_turns) = fork_turns {
            args["fork_turns"] = json!(fork_turns);
        }

        let output = SpawnAgentHandlerV2::default()
            .handle(invocation(
                session.clone(),
                turn.clone(),
                "spawn_agent",
                function_payload(args),
            ))
            .await
            .expect("Gemini omitted or empty fork_turns should spawn without a full-history fork");
        let (content, _) = expect_text_output(output);
        let result: SpawnAgentResult =
            serde_json::from_str(&content).expect("spawn_agent result should be json");
        let child_thread_id = session
            .services
            .agent_control
            .resolve_agent_reference(
                session.thread_id,
                &turn.session_source,
                result.task_name.as_str(),
            )
            .await
            .expect("spawned task name should resolve");
        let child_thread = manager
            .get_thread(child_thread_id)
            .await
            .expect("spawned child thread should exist");
        let snapshot = child_thread.config_snapshot().await;
        assert_eq!(snapshot.model, "gpt-5.4");
        assert_eq!(snapshot.reasoning_effort, Some(ReasoningEffort::Low));

        let child_history = child_thread.codex.session.clone_history().await;
        let child_history_json =
            serde_json::to_string(child_history.raw_items()).expect("serialize child history");
        assert!(
            !child_history_json.contains(parent_marker),
            "Gemini default-scoped child must not inherit parent history for {task_name}: {child_history_json}"
        );
        assert!(manager.captured_ops().iter().any(|(id, op)| {
            *id == child_thread_id
                && matches!(
                    op,
                    Op::InterAgentCommunication { communication }
                        if communication.author == AgentPath::root()
                            && communication.recipient.as_str() == result.task_name.as_str()
                            && communication.other_recipients.is_empty()
                            && communication.content == child_message
                            && communication.trigger_turn
                )
        }));
    }
}

#[tokio::test]
async fn multi_agent_v2_spawn_gemini_root_auto_arms_goal_once() {
    let setup = auto_goal_spawn_setup(
        AutoGoalProvider::Gemini,
        |_| SessionSource::Exec,
        Some("  Review   all worker results  "),
    )
    .await;

    spawn_agent_v2_for_auto_goal(&setup, "worker_a").await;

    let goal = setup
        .state_db
        .thread_goals()
        .get_thread_goal(setup.session.thread_id)
        .await
        .expect("thread goal read should succeed")
        .expect("Gemini root spawn should auto-arm a goal");
    assert_eq!(goal.objective, "Review all worker results");
    assert_eq!(goal.status, codex_state::ThreadGoalStatus::Active);
    assert_eq!(goal.token_budget, Some(250_000));

    spawn_agent_v2_for_auto_goal(&setup, "worker_b").await;

    let goal_after_second_spawn = setup
        .state_db
        .thread_goals()
        .get_thread_goal(setup.session.thread_id)
        .await
        .expect("thread goal read should succeed");
    assert_eq!(goal_after_second_spawn, Some(goal));
}

#[tokio::test]
async fn multi_agent_v2_spawn_non_gemini_does_not_auto_arm_goal() {
    let setup = auto_goal_spawn_setup(
        AutoGoalProvider::Bedrock,
        |_| SessionSource::Exec,
        Some("Review all worker results"),
    )
    .await;

    spawn_agent_v2_for_auto_goal(&setup, "worker").await;

    assert_eq!(
        None,
        setup
            .session
            .get_thread_goal()
            .await
            .expect("thread goal read should succeed")
    );
}

#[tokio::test]
async fn multi_agent_v2_spawn_gemini_auto_goal_respects_root_source_gate() {
    let cases = [
        (
            "internal",
            SessionSource::Internal(InternalSessionSource::MemoryConsolidation),
            false,
        ),
        (
            "subagent_no_agent_path",
            thread_spawn_source_for_test(ThreadId::new(), /*agent_path*/ None),
            false,
        ),
        (
            "subagent_root_agent_path",
            thread_spawn_source_for_test(ThreadId::new(), Some(AgentPath::root())),
            true,
        ),
        (
            "subagent_non_root_agent_path",
            thread_spawn_source_for_test(
                ThreadId::new(),
                Some(AgentPath::try_from("/root/child").expect("valid agent path")),
            ),
            false,
        ),
    ];

    for (label, session_source, expect_goal) in cases {
        let setup = auto_goal_spawn_setup(
            AutoGoalProvider::Gemini,
            |_| session_source,
            /*first_user_message*/ None,
        )
        .await;

        spawn_agent_v2_for_auto_goal(&setup, label).await;

        let goal = setup
            .session
            .get_thread_goal()
            .await
            .expect("thread goal read should succeed");
        assert_eq!(goal.is_some(), expect_goal, "{label}");
    }
}

#[tokio::test]
async fn multi_agent_v2_spawn_gemini_auto_goal_complete_stops_idle_continuation() {
    let setup = auto_goal_spawn_setup(
        AutoGoalProvider::Gemini,
        |_| SessionSource::Exec,
        Some("Review all worker results"),
    )
    .await;
    spawn_agent_v2_for_auto_goal(&setup, "worker").await;

    UpdateGoalHandler
        .handle(invocation(
            setup.session.clone(),
            setup.turn.clone(),
            "update_goal",
            function_payload(json!({"status": "complete"})),
        ))
        .await
        .expect("update_goal should mark the auto-goal complete");

    let goal = setup
        .session
        .get_thread_goal()
        .await
        .expect("thread goal read should succeed")
        .expect("auto-goal should remain persisted");
    assert_eq!(
        goal.status,
        codex_protocol::protocol::ThreadGoalStatus::Complete
    );

    setup
        .session
        .goal_runtime_apply(GoalRuntimeEvent::MaybeContinueIfIdle)
        .await
        .expect("idle goal runtime should apply");
    assert!(setup.session.active_turn.lock().await.is_none());
}

// --- Gemini orchestration structural auto-completion (Layer 1) -------------

/// Arms an Active orchestration goal on the setup's session, optionally marking
/// it as auto-armed (the marker the real Gemini auto-arm records). Mirrors the
/// auto-arm so child status can be driven deterministically.
async fn arm_orchestration_goal(setup: &AutoGoalSpawnSetup, mark_auto_armed: bool) {
    setup
        .session
        .create_thread_goal(
            setup.turn.as_ref(),
            crate::goals::CreateGoalRequest {
                objective: "Review all worker results".to_string(),
                token_budget: Some(250_000),
            },
        )
        .await
        .expect("orchestration goal should arm");
    if mark_auto_armed {
        setup.session.mark_auto_armed_orchestration_goal().await;
    }
}

async fn orchestration_goal_status(
    setup: &AutoGoalSpawnSetup,
) -> codex_protocol::protocol::ThreadGoalStatus {
    setup
        .session
        .get_thread_goal()
        .await
        .expect("thread goal read should succeed")
        .expect("goal should remain persisted")
        .status
}

#[test]
fn is_terminal_with_delivery_excludes_null_completion() {
    use crate::agent::status::is_terminal_with_delivery;
    // Done with delivery: a report, or a determinate error/shutdown verdict.
    assert!(is_terminal_with_delivery(&AgentStatus::Completed(Some(
        "report".to_string()
    ))));
    assert!(is_terminal_with_delivery(&AgentStatus::Errored(
        "boom".to_string()
    )));
    assert!(is_terminal_with_delivery(&AgentStatus::Shutdown));
    // NOT done: `Completed(None)` (null/no report) is indeterminate and
    // recoverable, so it must be chased, not treated as delivered. (O27: this is
    // exactly `is_conservatively_terminal` minus `Completed(None)`.)
    assert!(!is_terminal_with_delivery(&AgentStatus::Completed(None)));
    // NOT done: the non-terminal states, including a mid-retry child (Running).
    assert!(!is_terminal_with_delivery(&AgentStatus::NotFound));
    assert!(!is_terminal_with_delivery(&AgentStatus::PendingInit));
    assert!(!is_terminal_with_delivery(&AgentStatus::Running));
    assert!(!is_terminal_with_delivery(&AgentStatus::Interrupted));
}

#[tokio::test]
async fn gemini_auto_arm_records_orchestration_goal_marker() {
    let setup = auto_goal_spawn_setup(
        AutoGoalProvider::Gemini,
        |_| SessionSource::Exec,
        Some("Review all worker results"),
    )
    .await;
    spawn_agent_v2_for_auto_goal(&setup, "worker").await;

    let state_goal = setup
        .state_db
        .thread_goals()
        .get_thread_goal(setup.session.thread_id)
        .await
        .expect("thread goal read should succeed")
        .expect("Gemini root spawn should auto-arm a goal");
    let marker = setup
        .session
        .goal_runtime
        .auto_armed_orchestration_goal_id
        .lock()
        .await
        .clone();
    assert_eq!(marker, Some(state_goal.goal_id));
}

#[tokio::test]
async fn orchestration_auto_complete_fires_on_clean_synthesis_turn() {
    let setup = auto_goal_spawn_setup(
        AutoGoalProvider::Gemini,
        |_| SessionSource::Exec,
        Some("Review all worker results"),
    )
    .await;
    arm_orchestration_goal(&setup, /*mark_auto_armed*/ true).await;
    let (child_id, ..) = spawn_worker(&setup.session, &setup.turn, "worker").await;
    complete_worker_turn(&setup.manager, child_id, "child result").await;

    setup
        .session
        .maybe_auto_complete_gemini_orchestration_goal(
            setup.turn.as_ref(),
            /*emitted_final_answer*/ true,
            /*reengaged_child_this_turn*/ false,
        )
        .await;

    assert_eq!(
        orchestration_goal_status(&setup).await,
        codex_protocol::protocol::ThreadGoalStatus::Complete
    );
}

#[tokio::test]
async fn orchestration_auto_complete_skips_while_child_running() {
    let setup = auto_goal_spawn_setup(
        AutoGoalProvider::Gemini,
        |_| SessionSource::Exec,
        Some("Review all worker results"),
    )
    .await;
    arm_orchestration_goal(&setup, /*mark_auto_armed*/ true).await;
    // Child queued but no turn driven: stays PendingInit (not terminal-with-
    // delivery), so the goal must remain Active and the existing continuation
    // backstop is preserved.
    let (child_id, ..) = spawn_worker(&setup.session, &setup.turn, "worker").await;
    assert!(
        !crate::agent::status::is_terminal_with_delivery(
            &setup
                .session
                .services
                .agent_control
                .get_status(child_id)
                .await
        ),
        "freshly-spawned worker should not be terminal-with-delivery"
    );

    setup
        .session
        .maybe_auto_complete_gemini_orchestration_goal(
            setup.turn.as_ref(),
            /*emitted_final_answer*/ true,
            /*reengaged_child_this_turn*/ false,
        )
        .await;

    assert_eq!(
        orchestration_goal_status(&setup).await,
        codex_protocol::protocol::ThreadGoalStatus::Active
    );
}

#[tokio::test]
async fn orchestration_auto_complete_skips_on_followup_turn() {
    // Hole A: even with every child terminal, a turn that re-engaged a child
    // (spawn_agent/followup_task/send_message) must not auto-complete.
    let setup = auto_goal_spawn_setup(
        AutoGoalProvider::Gemini,
        |_| SessionSource::Exec,
        Some("Review all worker results"),
    )
    .await;
    arm_orchestration_goal(&setup, /*mark_auto_armed*/ true).await;
    let (child_id, ..) = spawn_worker(&setup.session, &setup.turn, "worker").await;
    complete_worker_turn(&setup.manager, child_id, "child result").await;

    setup
        .session
        .maybe_auto_complete_gemini_orchestration_goal(
            setup.turn.as_ref(),
            /*emitted_final_answer*/ true,
            /*reengaged_child_this_turn*/ true,
        )
        .await;

    assert_eq!(
        orchestration_goal_status(&setup).await,
        codex_protocol::protocol::ThreadGoalStatus::Active
    );
}

#[tokio::test]
async fn orchestration_auto_complete_skips_when_any_child_still_running() {
    // Interim message while a sibling is still working: one child terminal, one
    // not — any non-terminal child keeps the goal Active.
    let setup = auto_goal_spawn_setup(
        AutoGoalProvider::Gemini,
        |_| SessionSource::Exec,
        Some("Review all worker results"),
    )
    .await;
    arm_orchestration_goal(&setup, /*mark_auto_armed*/ true).await;
    let (done_child, ..) = spawn_worker(&setup.session, &setup.turn, "done_worker").await;
    let (_running_child, ..) = spawn_worker(&setup.session, &setup.turn, "running_worker").await;
    complete_worker_turn(&setup.manager, done_child, "done result").await;

    setup
        .session
        .maybe_auto_complete_gemini_orchestration_goal(
            setup.turn.as_ref(),
            /*emitted_final_answer*/ true,
            /*reengaged_child_this_turn*/ false,
        )
        .await;

    assert_eq!(
        orchestration_goal_status(&setup).await,
        codex_protocol::protocol::ThreadGoalStatus::Active
    );
}

#[tokio::test]
async fn orchestration_auto_complete_never_completes_user_created_goal() {
    // A goal created without the auto-armed marker (e.g. user-created before the
    // first spawn) must never be auto-completed, even on an otherwise-clean turn.
    let setup = auto_goal_spawn_setup(
        AutoGoalProvider::Gemini,
        |_| SessionSource::Exec,
        Some("Review all worker results"),
    )
    .await;
    arm_orchestration_goal(&setup, /*mark_auto_armed*/ false).await;
    let (child_id, ..) = spawn_worker(&setup.session, &setup.turn, "worker").await;
    complete_worker_turn(&setup.manager, child_id, "child result").await;

    setup
        .session
        .maybe_auto_complete_gemini_orchestration_goal(
            setup.turn.as_ref(),
            /*emitted_final_answer*/ true,
            /*reengaged_child_this_turn*/ false,
        )
        .await;

    assert_eq!(
        orchestration_goal_status(&setup).await,
        codex_protocol::protocol::ThreadGoalStatus::Active
    );
}

#[tokio::test]
async fn orchestration_auto_complete_is_noop_on_non_gemini() {
    // Non-Gemini providers must be byte-identical: the helper short-circuits on
    // the wire-api gate even if a marked goal and terminal children exist.
    let setup = auto_goal_spawn_setup(
        AutoGoalProvider::Bedrock,
        |_| SessionSource::Exec,
        Some("Review all worker results"),
    )
    .await;
    arm_orchestration_goal(&setup, /*mark_auto_armed*/ true).await;
    let (child_id, ..) = spawn_worker(&setup.session, &setup.turn, "worker").await;
    complete_worker_turn(&setup.manager, child_id, "child result").await;

    setup
        .session
        .maybe_auto_complete_gemini_orchestration_goal(
            setup.turn.as_ref(),
            /*emitted_final_answer*/ true,
            /*reengaged_child_this_turn*/ false,
        )
        .await;

    assert_eq!(
        orchestration_goal_status(&setup).await,
        codex_protocol::protocol::ThreadGoalStatus::Active
    );
}

#[tokio::test]
async fn orchestration_auto_complete_fires_across_turns_via_durable_answer() {
    // O27 core fix: the parent emits its consolidated answer on the synthesis turn
    // while a child is still running (so that turn cannot complete), then a later
    // text-free idle continuation turn — with the child now terminal — completes
    // via the durable "answer emitted" marker. Under O26 (which required the
    // answer on the SAME idle turn) this stayed Active and ground the budget.
    let setup = auto_goal_spawn_setup(
        AutoGoalProvider::Gemini,
        |_| SessionSource::Exec,
        Some("Review all worker results"),
    )
    .await;
    arm_orchestration_goal(&setup, /*mark_auto_armed*/ true).await;
    let (child_id, ..) = spawn_worker(&setup.session, &setup.turn, "worker").await;

    // Synthesis turn: answer emitted, but the child is still PendingInit, so the
    // turn cannot complete — it only records the durable answer marker.
    setup
        .session
        .maybe_auto_complete_gemini_orchestration_goal(
            setup.turn.as_ref(),
            /*emitted_final_answer*/ true,
            /*reengaged_child_this_turn*/ false,
        )
        .await;
    assert_eq!(
        orchestration_goal_status(&setup).await,
        codex_protocol::protocol::ThreadGoalStatus::Active
    );

    // Child reports; a later text-free continuation turn (no fresh answer)
    // completes via the durable marker.
    complete_worker_turn(&setup.manager, child_id, "child result").await;
    setup
        .session
        .maybe_auto_complete_gemini_orchestration_goal(
            setup.turn.as_ref(),
            /*emitted_final_answer*/ false,
            /*reengaged_child_this_turn*/ false,
        )
        .await;
    assert_eq!(
        orchestration_goal_status(&setup).await,
        codex_protocol::protocol::ThreadGoalStatus::Complete
    );
}

#[tokio::test]
async fn orchestration_auto_complete_chases_null_child() {
    // O27: a child that finished with no report (`Completed(None)`) is NOT
    // terminal-with-delivery. Even with an answer emitted and no re-engagement,
    // the goal stays Active so the parent chases a real report instead of
    // finalizing with a missing section. (Under O26's `is_conservatively_terminal`
    // this would have wrongly completed.)
    let setup = auto_goal_spawn_setup(
        AutoGoalProvider::Gemini,
        |_| SessionSource::Exec,
        Some("Review all worker results"),
    )
    .await;
    arm_orchestration_goal(&setup, /*mark_auto_armed*/ true).await;
    let (child_id, ..) = spawn_worker(&setup.session, &setup.turn, "worker").await;
    complete_worker_turn_without_report(&setup.manager, child_id).await;

    setup
        .session
        .maybe_auto_complete_gemini_orchestration_goal(
            setup.turn.as_ref(),
            /*emitted_final_answer*/ true,
            /*reengaged_child_this_turn*/ false,
        )
        .await;

    assert_eq!(
        orchestration_goal_status(&setup).await,
        codex_protocol::protocol::ThreadGoalStatus::Active
    );
}

#[tokio::test]
async fn orchestration_auto_complete_completes_with_errored_child() {
    // O27: an errored child is a determinate terminal verdict (delivered), so a
    // clean synthesis turn completes — the orchestrator surfaces the error in its
    // answer rather than spinning on re-engagement.
    let setup = auto_goal_spawn_setup(
        AutoGoalProvider::Gemini,
        |_| SessionSource::Exec,
        Some("Review all worker results"),
    )
    .await;
    arm_orchestration_goal(&setup, /*mark_auto_armed*/ true).await;
    let (child_id, ..) = spawn_worker(&setup.session, &setup.turn, "worker").await;
    error_worker_turn(&setup.manager, child_id, "worker boom").await;

    setup
        .session
        .maybe_auto_complete_gemini_orchestration_goal(
            setup.turn.as_ref(),
            /*emitted_final_answer*/ true,
            /*reengaged_child_this_turn*/ false,
        )
        .await;

    assert_eq!(
        orchestration_goal_status(&setup).await,
        codex_protocol::protocol::ThreadGoalStatus::Complete
    );
}

#[tokio::test]
async fn orchestration_auto_complete_reengagement_clears_durable_answer() {
    // O27 hole A (durable): a synthesis turn records the answer marker, but a
    // subsequent re-engagement (spawn/followup/send) clears it, so a later clean
    // idle turn must NOT complete until a *fresh* answer is emitted — a re-opened
    // child cannot be finalized on a stale answer.
    let setup = auto_goal_spawn_setup(
        AutoGoalProvider::Gemini,
        |_| SessionSource::Exec,
        Some("Review all worker results"),
    )
    .await;
    arm_orchestration_goal(&setup, /*mark_auto_armed*/ true).await;
    let (child_id, ..) = spawn_worker(&setup.session, &setup.turn, "worker").await;

    // Synthesis turn while the child is still running: records the durable marker.
    setup
        .session
        .maybe_auto_complete_gemini_orchestration_goal(
            setup.turn.as_ref(),
            /*emitted_final_answer*/ true,
            /*reengaged_child_this_turn*/ false,
        )
        .await;

    // A re-engagement clears the durable marker (single chokepoint).
    setup
        .session
        .mark_reengaged_child_this_turn(setup.turn.as_ref())
        .await;

    // Child now reports; a clean idle turn with no fresh answer must NOT complete
    // because the durable marker was cleared by the re-engagement.
    complete_worker_turn(&setup.manager, child_id, "child result").await;
    setup
        .session
        .maybe_auto_complete_gemini_orchestration_goal(
            setup.turn.as_ref(),
            /*emitted_final_answer*/ false,
            /*reengaged_child_this_turn*/ false,
        )
        .await;

    assert_eq!(
        orchestration_goal_status(&setup).await,
        codex_protocol::protocol::ThreadGoalStatus::Active
    );
}

#[tokio::test]
async fn multi_agent_v2_spawn_gemini_auto_goal_budget_limit_stops_idle_continuation() {
    let setup = auto_goal_spawn_setup(
        AutoGoalProvider::Gemini,
        |_| SessionSource::Exec,
        Some("Review all worker results"),
    )
    .await;
    spawn_agent_v2_for_auto_goal(&setup, "worker").await;

    setup
        .state_db
        .thread_goals()
        .account_thread_goal_usage(
            setup.session.thread_id,
            /*time_delta_seconds*/ 0,
            /*token_delta*/ 250_000,
            codex_state::GoalAccountingMode::ActiveOnly,
            /*expected_goal_id*/ None,
        )
        .await
        .expect("goal accounting should apply");

    let goal = setup
        .session
        .get_thread_goal()
        .await
        .expect("thread goal read should succeed")
        .expect("auto-goal should remain persisted");
    assert_eq!(
        goal.status,
        codex_protocol::protocol::ThreadGoalStatus::BudgetLimited
    );
    assert_eq!(goal.tokens_used, 250_000);

    setup
        .session
        .goal_runtime_apply(GoalRuntimeEvent::MaybeContinueIfIdle)
        .await
        .expect("idle goal runtime should apply");
    assert!(setup.session.active_turn.lock().await.is_none());
}

#[tokio::test]
async fn spawn_agent_service_tier_override_validates_the_effective_child_model() {
    #[derive(Debug, Deserialize)]
    struct SpawnAgentResult {
        agent_id: String,
    }

    {
        let (mut session, turn) = make_session_and_context().await;
        let manager = thread_manager();
        let root = manager
            .start_thread((*turn.config).clone())
            .await
            .expect("root thread should start");
        session.services.agent_control = manager.agent_control();
        session.thread_id = root.thread_id;

        let output = SpawnAgentHandler::default()
            .handle(invocation(
                Arc::new(session),
                Arc::new(turn),
                "spawn_agent",
                function_payload(json!({
                    "message": "inspect this repo",
                    "model": "gpt-5.4",
                    "service_tier": ServiceTier::Fast.request_value()
                })),
            ))
            .await
            .expect("spawn_agent should accept a supported explicit service tier");
        let (content, _) = expect_text_output(output);
        let result: SpawnAgentResult =
            serde_json::from_str(&content).expect("spawn_agent result should be json");
        let snapshot = manager
            .get_thread(parse_agent_id(&result.agent_id))
            .await
            .expect("spawned agent thread should exist")
            .config_snapshot()
            .await;

        assert_eq!(
            snapshot.service_tier,
            Some(ServiceTier::Fast.request_value().to_string())
        );
    }

    {
        let (session, turn) = make_session_and_context().await;
        let err = SpawnAgentHandler::default()
            .handle(invocation(
                Arc::new(session),
                Arc::new(turn),
                "spawn_agent",
                function_payload(json!({
                    "message": "inspect this repo",
                    "model": "gpt-5.4",
                    "service_tier": "turbo"
                })),
            ))
            .await
            .err()
            .expect("unknown service tier should be rejected");

        assert_eq!(
            err,
            FunctionCallError::RespondToModel(
                "Service tier `turbo` is not supported for model `gpt-5.4`. Supported service tiers: priority"
                    .to_string()
            )
        );
    }

    {
        let (session, turn) = make_session_and_context().await;
        let err = SpawnAgentHandler::default()
            .handle(invocation(
                Arc::new(session),
                Arc::new(turn),
                "spawn_agent",
                function_payload(json!({
                    "message": "inspect this repo",
                    "model": "gpt-5.3-codex",
                    "service_tier": ServiceTier::Fast.request_value()
                })),
            ))
            .await
            .err()
            .expect("tier unsupported by the final child model should be rejected");

        assert_eq!(
            err,
            FunctionCallError::RespondToModel(
                "Service tier `priority` is not supported for model `gpt-5.3-codex`. Supported service tiers: none"
                    .to_string()
            )
        );
    }
}

#[tokio::test]
async fn spawn_agent_service_tier_inheritance_preserves_supported_or_configured_tiers() {
    #[derive(Debug, Deserialize)]
    struct SpawnAgentResult {
        agent_id: String,
    }

    {
        let (mut session, turn) = make_session_and_context().await;
        let mut turn = turn
            .with_model("gpt-5.4".to_string(), &session.services.models_manager)
            .await;
        let mut config = (*turn.config).clone();
        config.service_tier = Some(ServiceTier::Fast.request_value().to_string());
        turn.config = Arc::new(config);
        let manager = thread_manager();
        let root = manager
            .start_thread((*turn.config).clone())
            .await
            .expect("root thread should start");
        session.services.agent_control = manager.agent_control();
        session.thread_id = root.thread_id;

        let output = SpawnAgentHandler::default()
            .handle(invocation(
                Arc::new(session),
                Arc::new(turn),
                "spawn_agent",
                function_payload(json!({"message": "inspect this repo"})),
            ))
            .await
            .expect("spawn_agent should inherit a supported parent service tier");
        let (content, _) = expect_text_output(output);
        let result: SpawnAgentResult =
            serde_json::from_str(&content).expect("spawn_agent result should be json");
        let snapshot = manager
            .get_thread(parse_agent_id(&result.agent_id))
            .await
            .expect("spawned agent thread should exist")
            .config_snapshot()
            .await;

        assert_eq!(
            snapshot.service_tier,
            Some(ServiceTier::Fast.request_value().to_string())
        );
    }

    {
        let (mut session, turn) = make_session_and_context().await;
        let mut turn = turn
            .with_model("gpt-5.4".to_string(), &session.services.models_manager)
            .await;
        let mut config = (*turn.config).clone();
        config.service_tier = Some(ServiceTier::Fast.request_value().to_string());
        turn.config = Arc::new(config);
        let manager = thread_manager();
        let root = manager
            .start_thread((*turn.config).clone())
            .await
            .expect("root thread should start");
        session.services.agent_control = manager.agent_control();
        session.thread_id = root.thread_id;

        let output = SpawnAgentHandler::default()
            .handle(invocation(
                Arc::new(session),
                Arc::new(turn),
                "spawn_agent",
                function_payload(json!({
                    "message": "inspect this repo",
                    "model": "gpt-5.3-codex"
                })),
            ))
            .await
            .expect("spawn_agent should clear unsupported inherited service tier");
        let (content, _) = expect_text_output(output);
        let result: SpawnAgentResult =
            serde_json::from_str(&content).expect("spawn_agent result should be json");
        let snapshot = manager
            .get_thread(parse_agent_id(&result.agent_id))
            .await
            .expect("spawned agent thread should exist")
            .config_snapshot()
            .await;

        assert_eq!(snapshot.service_tier, None);
    }

    {
        let (mut session, mut turn) = make_session_and_context().await;
        tokio::fs::create_dir_all(&turn.config.codex_home)
            .await
            .expect("codex home should be created");
        let role_config_path = turn
            .config
            .codex_home
            .as_path()
            .join("service-tier-role.toml");
        tokio::fs::write(
            &role_config_path,
            r#"model = "gpt-5.4"
service_tier = "priority"
"#,
        )
        .await
        .expect("role config should be written");

        let role_name = "service-tier-role".to_string();
        let mut config = (*turn.config).clone();
        config.agent_roles.insert(
            role_name.clone(),
            AgentRoleConfig {
                description: Some("Role with a child service tier".to_string()),
                config_file: Some(role_config_path),
                nickname_candidates: None,
            },
        );
        turn.config = Arc::new(config);
        let manager = thread_manager();
        let root = manager
            .start_thread((*turn.config).clone())
            .await
            .expect("root thread should start");
        session.services.agent_control = manager.agent_control();
        session.thread_id = root.thread_id;

        let output = SpawnAgentHandler::default()
            .handle(invocation(
                Arc::new(session),
                Arc::new(turn),
                "spawn_agent",
                function_payload(json!({
                    "message": "inspect this repo",
                    "agent_type": role_name
                })),
            ))
            .await
            .expect("spawn_agent should preserve the child role service tier");
        let (content, _) = expect_text_output(output);
        let result: SpawnAgentResult =
            serde_json::from_str(&content).expect("spawn_agent result should be json");
        let snapshot = manager
            .get_thread(parse_agent_id(&result.agent_id))
            .await
            .expect("spawned agent thread should exist")
            .config_snapshot()
            .await;

        assert_eq!(
            snapshot.service_tier,
            Some(ServiceTier::Fast.request_value().to_string())
        );
    }
}

#[tokio::test]
async fn spawn_agent_role_service_tier_falls_back_to_supported_parent_tier() {
    #[derive(Debug, Deserialize)]
    struct SpawnAgentResult {
        agent_id: String,
    }

    let (mut session, turn) = make_session_and_context().await;
    let mut turn = turn
        .with_model("gpt-5.4".to_string(), &session.services.models_manager)
        .await;
    tokio::fs::create_dir_all(&turn.config.codex_home)
        .await
        .expect("codex home should be created");
    let role_config_path = turn.config.codex_home.as_path().join("tiered-role.toml");
    tokio::fs::write(
        &role_config_path,
        r#"model = "gpt-5.4"
service_tier = "turbo"
"#,
    )
    .await
    .expect("role config should be written");

    let role_name = "tiered-role".to_string();
    let mut config = (*turn.config).clone();
    config.service_tier = Some(ServiceTier::Fast.request_value().to_string());
    config.agent_roles.insert(
        role_name.clone(),
        AgentRoleConfig {
            description: Some("Role with an unsupported child tier".to_string()),
            config_file: Some(role_config_path),
            nickname_candidates: None,
        },
    );
    turn.config = Arc::new(config);
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;

    let output = SpawnAgentHandler::default()
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "agent_type": role_name
            })),
        ))
        .await
        .expect("spawn_agent should fall back to the supported parent tier");
    let (content, _) = expect_text_output(output);
    let result: SpawnAgentResult =
        serde_json::from_str(&content).expect("spawn_agent result should be json");
    let snapshot = manager
        .get_thread(parse_agent_id(&result.agent_id))
        .await
        .expect("spawned agent thread should exist")
        .config_snapshot()
        .await;

    assert_eq!(
        snapshot.service_tier,
        Some(ServiceTier::Fast.request_value().to_string())
    );
}

#[tokio::test]
async fn spawn_agent_role_service_tier_does_not_hide_invalid_spawn_request() {
    let (session, mut turn) = make_session_and_context().await;
    tokio::fs::create_dir_all(&turn.config.codex_home)
        .await
        .expect("codex home should be created");
    let role_config_path = turn.config.codex_home.as_path().join("tiered-role.toml");
    tokio::fs::write(
        &role_config_path,
        r#"model = "gpt-5.4"
service_tier = "priority"
"#,
    )
    .await
    .expect("role config should be written");

    let role_name = "tiered-role".to_string();
    let mut config = (*turn.config).clone();
    config.agent_roles.insert(
        role_name.clone(),
        AgentRoleConfig {
            description: Some("Role with a supported child tier".to_string()),
            config_file: Some(role_config_path),
            nickname_candidates: None,
        },
    );
    turn.config = Arc::new(config);

    let result = SpawnAgentHandler::default()
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "agent_type": role_name,
                "service_tier": "turbo"
            })),
        ))
        .await;

    assert_eq!(
        result.err(),
        Some(FunctionCallError::RespondToModel(
            "Service tier `turbo` is not supported for model `gpt-5.4`. Supported service tiers: priority"
                .to_string()
        ))
    );
}

#[tokio::test]
async fn spawn_agent_full_history_fork_accepts_explicit_service_tier() {
    #[derive(Debug, Deserialize)]
    struct SpawnAgentResult {
        agent_id: String,
    }

    let (mut session, turn) = make_session_and_context().await;
    let turn = turn
        .with_model("gpt-5.4".to_string(), &session.services.models_manager)
        .await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;

    let output = SpawnAgentHandler::default()
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "fork_context": true,
                "service_tier": ServiceTier::Fast.request_value()
            })),
        ))
        .await
        .expect("full-history fork should accept explicit service tier");
    let (content, _) = expect_text_output(output);
    let result: SpawnAgentResult =
        serde_json::from_str(&content).expect("spawn_agent result should be json");
    let snapshot = manager
        .get_thread(parse_agent_id(&result.agent_id))
        .await
        .expect("spawned agent thread should exist")
        .config_snapshot()
        .await;

    assert_eq!(
        snapshot.service_tier,
        Some(ServiceTier::Fast.request_value().to_string())
    );
}

#[tokio::test]
async fn multi_agent_v2_full_history_fork_accepts_explicit_service_tier() {
    #[derive(Debug, Deserialize)]
    struct SpawnAgentResult {
        task_name: String,
    }

    let (mut session, turn) = make_session_and_context().await;
    let mut turn = turn
        .with_model("gpt-5.4".to_string(), &session.services.models_manager)
        .await;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    let output = SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "task_name": "fork_with_tier",
                "service_tier": ServiceTier::Fast.request_value(),
                "fork_turns": "all"
            })),
        ))
        .await
        .expect("multi-agent v2 full-history fork should accept explicit service tier");
    let (content, _) = expect_text_output(output);
    let result: SpawnAgentResult =
        serde_json::from_str(&content).expect("spawn_agent result should be json");
    let child_thread_id = session
        .services
        .agent_control
        .resolve_agent_reference(
            session.thread_id,
            &turn.session_source,
            result.task_name.as_str(),
        )
        .await
        .expect("spawned task name should resolve");
    let snapshot = manager
        .get_thread(child_thread_id)
        .await
        .expect("spawned agent thread should exist")
        .config_snapshot()
        .await;

    assert_eq!(
        snapshot.service_tier,
        Some(ServiceTier::Fast.request_value().to_string())
    );
}

#[tokio::test]
async fn multi_agent_v2_spawn_partial_fork_turns_allows_agent_type_override() {
    let (mut session, mut turn) = make_session_and_context().await;
    let role_name = install_role_with_model_override(&mut turn).await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    let turn = TurnContext {
        config: Arc::new(config),
        multi_agent_version: codex_protocol::protocol::MultiAgentVersion::V2,
        ..turn
    };

    let output = SpawnAgentHandlerV2::default()
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "task_name": "partial_fork",
                "agent_type": role_name,
                "fork_turns": "1"
            })),
        ))
        .await
        .expect("partial fork should allow agent_type overrides");
    let (content, _) = expect_text_output(output);
    let result: serde_json::Value =
        serde_json::from_str(&content).expect("spawn_agent result should be json");
    assert_eq!(result["task_name"], "/root/partial_fork");
    let agent_id = manager
        .captured_ops()
        .into_iter()
        .map(|(thread_id, _)| thread_id)
        .find(|thread_id| *thread_id != root.thread_id)
        .expect("spawned agent should receive an op");
    let snapshot = manager
        .get_thread(agent_id)
        .await
        .expect("spawned agent thread should exist")
        .config_snapshot()
        .await;

    assert_eq!(snapshot.model, "gpt-5-role-override");
    assert_eq!(snapshot.model_provider_id, "ollama");
    assert_eq!(snapshot.reasoning_effort, Some(ReasoningEffort::Minimal));
}

#[tokio::test]
async fn spawn_agent_returns_agent_id_without_task_name() {
    let (mut session, turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();

    let output = SpawnAgentHandler::default()
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo"
            })),
        ))
        .await
        .expect("spawn_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: serde_json::Value =
        serde_json::from_str(&content).expect("spawn_agent result should be json");

    assert!(result["agent_id"].is_string());
    assert!(result.get("task_name").is_none());
    assert!(result.get("nickname").is_some());
    assert_eq!(success, Some(true));
}

#[tokio::test]
async fn multi_agent_v2_spawn_requires_task_name() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);

    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "spawn_agent",
        function_payload(json!({
            "message": "inspect this repo"
        })),
    );
    let Err(err) = SpawnAgentHandlerV2::default().handle(invocation).await else {
        panic!("missing task_name should be rejected");
    };
    let FunctionCallError::RespondToModel(message) = err else {
        panic!("missing task_name should surface as a model-facing error");
    };
    assert!(message.contains("missing field `task_name`"));
}

#[tokio::test]
async fn multi_agent_v2_spawn_rejects_legacy_items_field() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);

    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "spawn_agent",
        function_payload(json!({
            "message": "inspect this repo",
            "items": [{"type": "text", "text": "inspect this repo"}],
            "task_name": "worker"
        })),
    );
    let Err(err) = SpawnAgentHandlerV2::default().handle(invocation).await else {
        panic!("legacy items field should be rejected");
    };
    let FunctionCallError::RespondToModel(message) = err else {
        panic!("legacy items field should surface as a model-facing error");
    };
    assert!(message.contains("unknown field `items`"));
}

#[tokio::test]
async fn spawn_agent_errors_when_manager_dropped() {
    let (session, turn) = make_session_and_context().await;
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "spawn_agent",
        function_payload(json!({"message": "hello"})),
    );
    let Err(err) = SpawnAgentHandler::default().handle(invocation).await else {
        panic!("spawn should fail without a manager");
    };
    assert_eq!(
        err,
        FunctionCallError::RespondToModel("collab manager unavailable".to_string())
    );
}

#[tokio::test]
async fn multi_agent_v2_spawn_returns_path_and_send_message_accepts_relative_path() {
    #[derive(Debug, Deserialize)]
    struct SpawnAgentResult {
        task_name: String,
        nickname: Option<String>,
    }

    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);

    let session = Arc::new(session);
    let turn = Arc::new(turn);
    let spawn_output = SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "task_name": "test_process"
            })),
        ))
        .await
        .expect("spawn_agent should succeed");
    let (content, _) = expect_text_output(spawn_output);
    let spawn_result: SpawnAgentResult =
        serde_json::from_str(&content).expect("spawn result should parse");
    assert_eq!(spawn_result.task_name, "/root/test_process");
    assert_eq!(spawn_result.nickname, None);

    let child_thread_id = session
        .services
        .agent_control
        .resolve_agent_reference(session.thread_id, &turn.session_source, "test_process")
        .await
        .expect("relative path should resolve");
    let child_snapshot = manager
        .get_thread(child_thread_id)
        .await
        .expect("child thread should exist")
        .config_snapshot()
        .await;
    assert_eq!(
        child_snapshot.session_source.get_agent_path().as_deref(),
        Some("/root/test_process")
    );
    assert!(manager.captured_ops().iter().any(|(id, op)| {
        *id == child_thread_id
            && matches!(
                op,
                Op::InterAgentCommunication { communication }
                    if communication.author == AgentPath::root()
                        && communication.recipient.as_str() == "/root/test_process"
                        && communication.other_recipients.is_empty()
                        && communication.content == "inspect this repo"
                        && communication.trigger_turn
            )
    }));

    SendMessageHandlerV2
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "send_message",
            function_payload(json!({
                "target": "test_process",
                "message": "continue"
            })),
        ))
        .await
        .expect("send_message should accept v2 path");

    assert!(manager.captured_ops().iter().any(|(id, op)| {
        *id == child_thread_id
            && matches!(
                op,
                Op::InterAgentCommunication { communication }
                    if communication.author == AgentPath::root()
                        && communication.recipient.as_str() == "/root/test_process"
                        && communication.other_recipients.is_empty()
                        && communication.content == "continue"
                        && !communication.trigger_turn
            )
    }));
}

#[tokio::test]
async fn multi_agent_v2_spawn_rejects_legacy_fork_context() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);

    let err = SpawnAgentHandlerV2::default()
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "task_name": "worker",
                "fork_context": true
            })),
        ))
        .await
        .err()
        .expect("legacy fork_context should be rejected");

    assert_eq!(
        err,
        FunctionCallError::RespondToModel(
            "fork_context is not supported in MultiAgentV2; use fork_turns instead".to_string()
        )
    );
}

#[tokio::test]
async fn multi_agent_v2_spawn_rejects_invalid_fork_turns_string() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);

    let err = SpawnAgentHandlerV2::default()
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "task_name": "worker",
                "fork_turns": "banana"
            })),
        ))
        .await
        .err()
        .expect("invalid fork_turns should be rejected");

    assert_eq!(
        err,
        FunctionCallError::RespondToModel(
            "fork_turns must be `none`, `all`, or a positive integer string".to_string()
        )
    );
}

#[tokio::test]
async fn multi_agent_v2_spawn_rejects_zero_fork_turns() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);

    let err = SpawnAgentHandlerV2::default()
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "task_name": "worker",
                "fork_turns": "0"
            })),
        ))
        .await
        .err()
        .expect("zero turn count should be rejected");

    assert_eq!(
        err,
        FunctionCallError::RespondToModel(
            "fork_turns must be `none`, `all`, or a positive integer string".to_string()
        )
    );
}

#[tokio::test]
async fn multi_agent_v2_send_message_accepts_root_target_from_child() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;

    let child_path = AgentPath::try_from("/root/worker").expect("agent path");
    let child_thread_id = session
        .services
        .agent_control
        .spawn_agent_with_metadata(
            (*turn.config).clone(),
            vec![UserInput::Text {
                text: "inspect this repo".to_string(),
                text_elements: Vec::new(),
            }]
            .into(),
            Some(SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
                parent_thread_id: root.thread_id,
                depth: 1,
                agent_path: Some(child_path.clone()),
                agent_nickname: None,
                agent_role: None,
            })),
            crate::agent::control::SpawnAgentOptions::default(),
        )
        .await
        .expect("worker spawn should succeed")
        .thread_id;
    session.thread_id = child_thread_id;
    turn.session_source = SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
        parent_thread_id: root.thread_id,
        depth: 1,
        agent_path: Some(child_path.clone()),
        agent_nickname: None,
        agent_role: None,
    });

    let output = SendMessageHandlerV2
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "send_message",
            function_payload(json!({
                "target": "/root",
                "message": "done"
            })),
        ))
        .await
        .expect("send_message should accept the root agent path");
    assert_eq!(expect_text_output(output), (String::new(), Some(true)));

    assert!(manager.captured_ops().iter().any(|(id, op)| {
        *id == root.thread_id
            && matches!(
                op,
                Op::InterAgentCommunication { communication }
                    if communication.author == child_path
                        && communication.recipient == AgentPath::root()
                        && communication.other_recipients.is_empty()
                        && communication.content == "done"
                        && !communication.trigger_turn
            )
    }));
}

#[tokio::test]
async fn multi_agent_v2_followup_task_rejects_root_target_from_child() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;

    let child_path = AgentPath::try_from("/root/worker").expect("agent path");
    let child_thread_id = session
        .services
        .agent_control
        .spawn_agent_with_metadata(
            (*turn.config).clone(),
            vec![UserInput::Text {
                text: "inspect this repo".to_string(),
                text_elements: Vec::new(),
            }]
            .into(),
            Some(SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
                parent_thread_id: root.thread_id,
                depth: 1,
                agent_path: Some(child_path.clone()),
                agent_nickname: None,
                agent_role: None,
            })),
            crate::agent::control::SpawnAgentOptions::default(),
        )
        .await
        .expect("worker spawn should succeed")
        .thread_id;
    session.thread_id = child_thread_id;
    turn.session_source = SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
        parent_thread_id: root.thread_id,
        depth: 1,
        agent_path: Some(child_path),
        agent_nickname: None,
        agent_role: None,
    });

    let Err(err) = FollowupTaskHandlerV2
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "followup_task",
            function_payload(json!({
                "target": "/root",
                "message": "run this",
            })),
        ))
        .await
    else {
        panic!("followup_task should reject the root target");
    };

    assert_eq!(
        err,
        FunctionCallError::RespondToModel(
            "Follow-up tasks can't target the root agent".to_string()
        )
    );
    let root_ops = manager
        .captured_ops()
        .into_iter()
        .filter_map(|(id, op)| (id == root.thread_id).then_some(op))
        .collect::<Vec<_>>();
    assert!(!root_ops.iter().any(|op| matches!(op, Op::Interrupt)));
    assert!(
        !root_ops
            .iter()
            .any(|op| matches!(op, Op::InterAgentCommunication { .. }))
    );
}

#[tokio::test]
async fn multi_agent_v2_list_agents_returns_completed_status_and_last_task_message() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    let _ = config.features.enable(Feature::MultiAgentV2);
    set_turn_config(&mut turn, config);

    let session = Arc::new(session);
    let turn = Arc::new(turn);
    let spawn_output = SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "task_name": "worker"
            })),
        ))
        .await
        .expect("spawn_agent should succeed");
    let _ = expect_text_output(spawn_output);

    let agent_id = session
        .services
        .agent_control
        .resolve_agent_reference(session.thread_id, &turn.session_source, "worker")
        .await
        .expect("worker path should resolve");
    let child_thread = manager
        .get_thread(agent_id)
        .await
        .expect("child thread should exist");
    let child_turn = child_thread.codex.session.new_default_turn().await;
    child_thread
        .codex
        .session
        .send_event(
            child_turn.as_ref(),
            EventMsg::TurnComplete(TurnCompleteEvent {
                turn_id: child_turn.sub_id.clone(),
                last_agent_message: Some("done".to_string()),
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            }),
        )
        .await;

    let output = ListAgentsHandlerV2
        .handle(invocation(
            session,
            turn,
            "list_agents",
            function_payload(json!({})),
        ))
        .await
        .expect("list_agents should succeed");
    let (content, success) = expect_text_output(output);
    let result: ListAgentsResult =
        serde_json::from_str(&content).expect("list_agents result should be json");

    let agent_names = result
        .agents
        .iter()
        .map(|agent| agent.agent_name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(agent_names, vec!["/root", "/root/worker"]);
    let root_agent = result
        .agents
        .iter()
        .find(|agent| agent.agent_name == "/root")
        .expect("root agent should be listed");
    assert_eq!(root_agent.last_task_message.as_deref(), Some("Main thread"));
    let worker = result
        .agents
        .iter()
        .find(|agent| agent.agent_name == "/root/worker")
        .expect("worker agent should be listed");
    assert_eq!(worker.agent_status, json!({"completed": "done"}));
    assert_eq!(
        worker.last_task_message.as_deref(),
        Some("inspect this repo")
    );
    assert_eq!(success, Some(true));
}

#[tokio::test]
async fn multi_agent_v2_list_agents_filters_by_relative_path_prefix() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let mut config = (*turn.config).clone();
    let _ = config.features.enable(Feature::MultiAgentV2);
    set_turn_config(&mut turn, config.clone());
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;

    let researcher_path = AgentPath::from_string("/root/researcher".to_string()).expect("path");
    let worker_path = AgentPath::from_string("/root/researcher/worker".to_string()).expect("path");
    session
        .services
        .agent_control
        .spawn_agent_with_metadata(
            config.clone(),
            vec![UserInput::Text {
                text: "research".to_string(),
                text_elements: Vec::new(),
            }]
            .into(),
            Some(SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
                parent_thread_id: root.thread_id,
                depth: 1,
                agent_path: Some(researcher_path.clone()),
                agent_nickname: None,
                agent_role: None,
            })),
            crate::agent::control::SpawnAgentOptions::default(),
        )
        .await
        .expect("researcher agent should spawn");
    session
        .services
        .agent_control
        .spawn_agent_with_metadata(
            config,
            vec![UserInput::Text {
                text: "build".to_string(),
                text_elements: Vec::new(),
            }]
            .into(),
            Some(SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
                parent_thread_id: root.thread_id,
                depth: 2,
                agent_path: Some(worker_path.clone()),
                agent_nickname: None,
                agent_role: None,
            })),
            crate::agent::control::SpawnAgentOptions::default(),
        )
        .await
        .expect("worker agent should spawn");

    turn.session_source = SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
        parent_thread_id: root.thread_id,
        depth: 1,
        agent_path: Some(researcher_path),
        agent_nickname: None,
        agent_role: None,
    });

    let output = ListAgentsHandlerV2
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "list_agents",
            function_payload(json!({
                "path_prefix": "worker"
            })),
        ))
        .await
        .expect("list_agents should succeed");
    let (content, _) = expect_text_output(output);
    let result: ListAgentsResult =
        serde_json::from_str(&content).expect("list_agents result should be json");

    assert_eq!(result.agents.len(), 1);
    assert_eq!(result.agents[0].agent_name, worker_path.as_str());
    assert_eq!(result.agents[0].last_task_message.as_deref(), Some("build"));
}

#[tokio::test]
async fn multi_agent_v2_list_agents_omits_closed_agents() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    let _ = config.features.enable(Feature::MultiAgentV2);
    set_turn_config(&mut turn, config);

    let session = Arc::new(session);
    let turn = Arc::new(turn);
    let spawn_output = SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "task_name": "worker"
            })),
        ))
        .await
        .expect("spawn_agent should succeed");
    let _ = expect_text_output(spawn_output);

    let agent_id = session
        .services
        .agent_control
        .resolve_agent_reference(session.thread_id, &turn.session_source, "worker")
        .await
        .expect("worker path should resolve");
    session
        .services
        .agent_control
        .close_agent(agent_id)
        .await
        .expect("close_agent should succeed");

    let output = ListAgentsHandlerV2
        .handle(invocation(
            session,
            turn,
            "list_agents",
            function_payload(json!({})),
        ))
        .await
        .expect("list_agents should succeed");
    let (content, _) = expect_text_output(output);
    let result: ListAgentsResult =
        serde_json::from_str(&content).expect("list_agents result should be json");

    assert_eq!(result.agents.len(), 1);
    assert_eq!(result.agents[0].agent_name, "/root");
    assert_eq!(
        result.agents[0].last_task_message.as_deref(),
        Some("Main thread")
    );
}

#[tokio::test]
async fn multi_agent_v2_send_message_rejects_legacy_items_field() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = turn.config.as_ref().clone();
    let _ = config.features.enable(Feature::MultiAgentV2);
    set_turn_config(&mut turn, config);
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "boot worker",
                "task_name": "worker"
            })),
        ))
        .await
        .expect("spawn worker");
    let agent_id = session
        .services
        .agent_control
        .resolve_agent_reference(session.thread_id, &turn.session_source, "worker")
        .await
        .expect("worker should resolve");
    let invocation = invocation(
        session,
        turn,
        "send_message",
        function_payload(json!({
            "target": agent_id.to_string(),
            "items": [
                {"type": "mention", "name": "drive", "path": "app://google_drive"},
                {"type": "text", "text": "read the folder"}
            ]
        })),
    );

    let Err(err) = SendMessageHandlerV2.handle(invocation).await else {
        panic!("legacy items field should be rejected in v2");
    };
    let FunctionCallError::RespondToModel(message) = err else {
        panic!("legacy items field should surface as a model-facing error");
    };
    assert!(message.contains("unknown field `items`"));
}

#[tokio::test]
async fn multi_agent_v2_send_message_rejects_interrupt_parameter() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = turn.config.as_ref().clone();
    let _ = config.features.enable(Feature::MultiAgentV2);
    set_turn_config(&mut turn, config);
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "boot worker",
                "task_name": "worker"
            })),
        ))
        .await
        .expect("spawn worker");
    let agent_id = session
        .services
        .agent_control
        .resolve_agent_reference(session.thread_id, &turn.session_source, "worker")
        .await
        .expect("worker should resolve");

    let invocation = invocation(
        session,
        turn,
        "send_message",
        function_payload(json!({
            "target": agent_id.to_string(),
            "message": "continue",
            "interrupt": true
        })),
    );

    let Err(err) = SendMessageHandlerV2.handle(invocation).await else {
        panic!("send_message interrupt parameter should be rejected");
    };
    let FunctionCallError::RespondToModel(message) = err else {
        panic!("expected model-facing parse error");
    };
    assert!(message.starts_with(
        "failed to parse function arguments: unknown field `interrupt`, expected `target` or `message`"
    ));

    let ops = manager.captured_ops();
    let ops_for_agent: Vec<&Op> = ops
        .iter()
        .filter_map(|(id, op)| (*id == agent_id).then_some(op))
        .collect();
    assert!(!ops_for_agent.iter().any(|op| matches!(op, Op::Interrupt)));
    assert!(!ops_for_agent.iter().any(|op| matches!(
        op,
        Op::InterAgentCommunication { communication }
            if communication.author == AgentPath::root()
                && communication.recipient.as_str() == "/root/worker"
                && communication.other_recipients.is_empty()
                && communication.content == "continue"
                && !communication.trigger_turn
    )));
}

#[tokio::test]
async fn multi_agent_v2_followup_task_completion_notifies_parent_on_every_turn() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let mut config = turn.config.as_ref().clone();
    let _ = config.features.enable(Feature::MultiAgentV2);
    set_turn_config(&mut turn, config);
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    // Production spawn_agent calls happen after the parent turn has resolved
    // and stored its runtime; mirror that before using the synthetic handler.
    root.thread.codex.session.new_default_turn().await;
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "boot worker",
                "task_name": "worker"
            })),
        ))
        .await
        .expect("spawn worker");
    let agent_id = session
        .services
        .agent_control
        .resolve_agent_reference(session.thread_id, &turn.session_source, "worker")
        .await
        .expect("worker should resolve");
    let thread = manager
        .get_thread(agent_id)
        .await
        .expect("worker thread should exist");
    let worker_path = AgentPath::try_from("/root/worker").expect("worker path");

    let first_turn = thread.codex.session.new_default_turn().await;
    thread
        .codex
        .session
        .send_event(
            first_turn.as_ref(),
            EventMsg::TurnComplete(TurnCompleteEvent {
                turn_id: first_turn.sub_id.clone(),
                last_agent_message: Some("first done".to_string()),
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            }),
        )
        .await;

    FollowupTaskHandlerV2
        .handle(invocation(
            session,
            turn,
            "followup_task",
            function_payload(json!({
                "target": agent_id.to_string(),
                "message": "continue",
            })),
        ))
        .await
        .expect("followup_task should succeed");

    let second_turn = thread.codex.session.new_default_turn().await;
    thread
        .codex
        .session
        .send_event(
            second_turn.as_ref(),
            EventMsg::TurnComplete(TurnCompleteEvent {
                turn_id: second_turn.sub_id.clone(),
                last_agent_message: Some("second done".to_string()),
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            }),
        )
        .await;

    let first_notification = format_subagent_notification_message(
        worker_path.as_str(),
        &AgentStatus::Completed(Some("first done".to_string())),
    );
    let second_notification = format_subagent_notification_message(
        worker_path.as_str(),
        &AgentStatus::Completed(Some("second done".to_string())),
    );

    let notifications = timeout(Duration::from_secs(5), async {
        loop {
            let notifications = manager
                .captured_ops()
                .into_iter()
                .filter_map(|(id, op)| {
                    (id == root.thread_id)
                        .then_some(op)
                        .and_then(|op| match op {
                            Op::InterAgentCommunication { communication }
                                if communication.author == worker_path
                                    && communication.recipient == AgentPath::root()
                                    && communication.other_recipients.is_empty()
                                    && !communication.trigger_turn =>
                            {
                                Some(communication.content)
                            }
                            _ => None,
                        })
                })
                .collect::<Vec<_>>();
            let first_count = notifications
                .iter()
                .filter(|message| **message == first_notification)
                .count();
            let second_count = notifications
                .iter()
                .filter(|message| **message == second_notification)
                .count();
            if first_count == 1 && second_count == 1 {
                break notifications;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("parent should receive one completion notification per child turn");

    assert_eq!(notifications.len(), 2);
}

#[tokio::test]
async fn multi_agent_v2_followup_task_rejects_legacy_items_field() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = turn.config.as_ref().clone();
    let _ = config.features.enable(Feature::MultiAgentV2);
    set_turn_config(&mut turn, config);
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "boot worker",
                "task_name": "worker"
            })),
        ))
        .await
        .expect("spawn worker");
    let agent_id = session
        .services
        .agent_control
        .resolve_agent_reference(session.thread_id, &turn.session_source, "worker")
        .await
        .expect("worker should resolve");
    let invocation = invocation(
        session,
        turn,
        "followup_task",
        function_payload(json!({
            "target": agent_id.to_string(),
            "items": [{"type": "text", "text": "continue"}],
        })),
    );

    let Err(err) = FollowupTaskHandlerV2.handle(invocation).await else {
        panic!("legacy items field should be rejected in v2");
    };
    let FunctionCallError::RespondToModel(message) = err else {
        panic!("legacy items field should surface as a model-facing error");
    };
    assert!(message.contains("unknown field `items`"));
}

#[tokio::test]
async fn multi_agent_v2_interrupted_turn_does_not_notify_parent() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = turn.config.as_ref().clone();
    let _ = config.features.enable(Feature::MultiAgentV2);
    set_turn_config(&mut turn, config);
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "boot worker",
                "task_name": "worker"
            })),
        ))
        .await
        .expect("spawn worker");
    let agent_id = session
        .services
        .agent_control
        .resolve_agent_reference(session.thread_id, &turn.session_source, "worker")
        .await
        .expect("worker should resolve");
    let thread = manager
        .get_thread(agent_id)
        .await
        .expect("worker thread should exist");

    let aborted_turn = thread.codex.session.new_default_turn().await;
    thread
        .codex
        .session
        .send_event(
            aborted_turn.as_ref(),
            EventMsg::TurnAborted(TurnAbortedEvent {
                turn_id: Some(aborted_turn.sub_id.clone()),
                reason: TurnAbortReason::Interrupted,
                completed_at: None,
                duration_ms: None,
            }),
        )
        .await;

    let notifications = manager
        .captured_ops()
        .into_iter()
        .filter_map(|(id, op)| {
            (id == root.thread_id)
                .then_some(op)
                .and_then(|op| match op {
                    Op::InterAgentCommunication { communication }
                        if communication.author.as_str() == "/root/worker"
                            && communication.recipient == AgentPath::root()
                            && communication.other_recipients.is_empty() =>
                    {
                        Some(communication.content)
                    }
                    _ => None,
                })
        })
        .collect::<Vec<_>>();

    assert_eq!(notifications, Vec::<String>::new());
}

#[tokio::test]
async fn multi_agent_v2_spawn_omits_agent_id_when_named() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);

    let output = SpawnAgentHandlerV2::default()
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "task_name": "test_process"
            })),
        ))
        .await
        .expect("spawn_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: serde_json::Value =
        serde_json::from_str(&content).expect("spawn_agent result should be json");

    assert!(result.get("agent_id").is_none());
    assert_eq!(result["task_name"], "/root/test_process");
    assert!(result.get("nickname").is_none());
    assert_eq!(success, Some(true));
}

#[tokio::test]
async fn multi_agent_v2_spawn_surfaces_task_name_validation_errors() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);

    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "spawn_agent",
        function_payload(json!({
            "message": "inspect this repo",
            "task_name": "BadName"
        })),
    );
    let Err(err) = SpawnAgentHandlerV2::default().handle(invocation).await else {
        panic!("invalid agent name should be rejected");
    };
    assert_eq!(
        err,
        FunctionCallError::RespondToModel(
            "agent_name must use only lowercase letters, digits, and underscores".to_string()
        )
    );
}

#[tokio::test]
async fn spawn_agent_reapplies_runtime_sandbox_after_role_config() {
    #[derive(Debug, Deserialize)]
    struct SpawnAgentResult {
        agent_id: String,
        nickname: Option<String>,
    }

    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();
    let expected_sandbox = turn.config.legacy_sandbox_policy();
    #[allow(deprecated)]
    let mut expected_file_system_sandbox_policy =
        FileSystemSandboxPolicy::from_legacy_sandbox_policy_for_cwd(&expected_sandbox, &turn.cwd);
    expected_file_system_sandbox_policy
        .entries
        .push(FileSystemSandboxEntry {
            path: FileSystemPath::GlobPattern {
                pattern: "**/.env".to_string(),
            },
            access: FileSystemAccessMode::Deny,
        });
    let expected_network_sandbox_policy = NetworkSandboxPolicy::from(&expected_sandbox);
    let expected_permission_profile = PermissionProfile::from_runtime_permissions_with_enforcement(
        SandboxEnforcement::from_legacy_sandbox_policy(&expected_sandbox),
        &expected_file_system_sandbox_policy,
        expected_network_sandbox_policy,
    );
    turn.approval_policy
        .set(AskForApproval::OnRequest)
        .expect("approval policy should be set");
    turn.permission_profile = expected_permission_profile.clone();
    assert_ne!(
        expected_permission_profile,
        turn.config.permissions.effective_permission_profile(),
        "test requires a runtime profile override that differs from base config"
    );

    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "spawn_agent",
        function_payload(json!({
            "message": "await this command",
            "agent_type": "explorer"
        })),
    );
    let output = SpawnAgentHandler::default()
        .handle(invocation)
        .await
        .expect("spawn_agent should succeed");
    let (content, _) = expect_text_output(output);
    let result: SpawnAgentResult =
        serde_json::from_str(&content).expect("spawn_agent result should be json");
    let agent_id = parse_agent_id(&result.agent_id);
    assert!(
        result
            .nickname
            .as_deref()
            .is_some_and(|nickname| !nickname.is_empty())
    );

    let snapshot = manager
        .get_thread(agent_id)
        .await
        .expect("spawned agent thread should exist")
        .config_snapshot()
        .await;
    assert_eq!(snapshot.sandbox_policy(), expected_sandbox);
    assert_eq!(snapshot.approval_policy, AskForApproval::OnRequest);
    assert_eq!(snapshot.permission_profile, expected_permission_profile);
    let child_thread = manager
        .get_thread(agent_id)
        .await
        .expect("spawned agent thread should exist");
    let child_turn = child_thread.codex.session.new_default_turn().await;
    assert_eq!(
        child_turn.file_system_sandbox_policy(),
        expected_file_system_sandbox_policy
    );
    assert_eq!(
        child_turn.network_sandbox_policy(),
        expected_network_sandbox_policy
    );
    assert_eq!(child_turn.permission_profile(), expected_permission_profile);
}

#[tokio::test]
async fn spawn_agent_rejects_when_depth_limit_exceeded() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();

    let max_depth = turn.config.agent_max_depth;
    turn.session_source = SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
        parent_thread_id: session.thread_id,
        depth: max_depth,
        agent_path: None,
        agent_nickname: None,
        agent_role: None,
    });

    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "spawn_agent",
        function_payload(json!({"message": "hello"})),
    );
    let Err(err) = SpawnAgentHandler::default().handle(invocation).await else {
        panic!("spawn should fail when depth limit exceeded");
    };
    assert_eq!(
        err,
        FunctionCallError::RespondToModel(
            "Agent depth limit reached. Solve the task yourself.".to_string()
        )
    );
}

#[tokio::test]
async fn spawn_agent_allows_depth_up_to_configured_max_depth() {
    #[derive(Debug, Deserialize)]
    struct SpawnAgentResult {
        agent_id: String,
        nickname: Option<String>,
    }

    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();

    let mut config = (*turn.config).clone();
    config.agent_max_depth = DEFAULT_AGENT_MAX_DEPTH + 1;
    turn.config = Arc::new(config);
    turn.session_source = SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
        parent_thread_id: session.thread_id,
        depth: DEFAULT_AGENT_MAX_DEPTH,
        agent_path: None,
        agent_nickname: None,
        agent_role: None,
    });

    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "spawn_agent",
        function_payload(json!({"message": "hello"})),
    );
    let output = SpawnAgentHandler::default()
        .handle(invocation)
        .await
        .expect("spawn should succeed within configured depth");
    let (content, success) = expect_text_output(output);
    let result: SpawnAgentResult =
        serde_json::from_str(&content).expect("spawn_agent result should be json");
    assert!(!result.agent_id.is_empty());
    assert!(
        result
            .nickname
            .as_deref()
            .is_some_and(|nickname| !nickname.is_empty())
    );
    assert_eq!(success, Some(true));
}

#[tokio::test]
async fn multi_agent_v2_spawn_agent_ignores_configured_max_depth() {
    #[derive(Debug, Deserialize)]
    struct SpawnAgentResult {
        task_name: String,
        nickname: Option<String>,
    }

    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let mut config = (*turn.config).clone();
    config.agent_max_depth = 1;
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    let root = manager
        .start_thread(config.clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    set_turn_config(&mut turn, config);
    let parent_path = AgentPath::try_from("/root/parent").expect("agent path");
    turn.session_source = SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
        parent_thread_id: root.thread_id,
        depth: 1,
        agent_path: Some(parent_path),
        agent_nickname: None,
        agent_role: None,
    });

    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "spawn_agent",
        function_payload(json!({
            "message": "hello",
            "task_name": "child",
            "fork_turns": "none"
        })),
    );
    let output = SpawnAgentHandlerV2::default()
        .handle(invocation)
        .await
        .expect("multi-agent v2 spawn should ignore max depth");
    let (content, success) = expect_text_output(output);
    let result: SpawnAgentResult =
        serde_json::from_str(&content).expect("spawn_agent result should be json");
    assert_eq!(result.task_name, "/root/parent/child");
    assert_eq!(result.nickname, None);
    assert_eq!(success, Some(true));
}

#[tokio::test]
async fn send_input_rejects_empty_message() {
    let (session, turn) = make_session_and_context().await;
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "send_input",
        function_payload(json!({"target": ThreadId::new().to_string(), "message": ""})),
    );
    let Err(err) = SendInputHandler.handle(invocation).await else {
        panic!("empty message should be rejected");
    };
    assert_eq!(
        err,
        FunctionCallError::RespondToModel("Empty message can't be sent to an agent".to_string())
    );
}

#[tokio::test]
async fn send_input_rejects_when_message_and_items_are_both_set() {
    let (session, turn) = make_session_and_context().await;
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "send_input",
        function_payload(json!({
            "target": ThreadId::new().to_string(),
            "message": "hello",
            "items": [{"type": "mention", "name": "drive", "path": "app://drive"}]
        })),
    );
    let Err(err) = SendInputHandler.handle(invocation).await else {
        panic!("message+items should be rejected");
    };
    assert_eq!(
        err,
        FunctionCallError::RespondToModel(
            "Provide either message or items, but not both".to_string()
        )
    );
}

#[tokio::test]
async fn send_input_rejects_invalid_id() {
    let (session, turn) = make_session_and_context().await;
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "send_input",
        function_payload(json!({"target": "not-a-uuid", "message": "hi"})),
    );
    let Err(err) = SendInputHandler.handle(invocation).await else {
        panic!("invalid id should be rejected");
    };
    let FunctionCallError::RespondToModel(msg) = err else {
        panic!("expected respond-to-model error");
    };
    assert!(msg.starts_with("invalid agent id not-a-uuid:"));
}

#[tokio::test]
async fn send_input_reports_missing_agent() {
    let (mut session, turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();
    let agent_id = ThreadId::new();
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "send_input",
        function_payload(json!({"target": agent_id.to_string(), "message": "hi"})),
    );
    let Err(err) = SendInputHandler.handle(invocation).await else {
        panic!("missing agent should be reported");
    };
    assert_eq!(
        err,
        FunctionCallError::RespondToModel(format!("agent with id {agent_id} not found"))
    );
}

#[tokio::test]
async fn send_input_interrupts_before_prompt() {
    let (mut session, turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();
    let config = turn.config.as_ref().clone();
    let thread = manager
        .start_thread(config.clone())
        .await
        .expect("start thread");
    let agent_id = thread.thread_id;
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "send_input",
        function_payload(json!({
            "target": agent_id.to_string(),
            "message": "hi",
            "interrupt": true
        })),
    );
    SendInputHandler
        .handle(invocation)
        .await
        .expect("send_input should succeed");

    let ops = manager.captured_ops();
    let ops_for_agent: Vec<&Op> = ops
        .iter()
        .filter_map(|(id, op)| (*id == agent_id).then_some(op))
        .collect();
    assert_eq!(ops_for_agent.len(), 2);
    assert!(matches!(ops_for_agent[0], Op::Interrupt));
    assert!(matches!(ops_for_agent[1], Op::UserInput { .. }));

    let _ = thread
        .thread
        .submit(Op::Shutdown {})
        .await
        .expect("shutdown should submit");
}

#[tokio::test]
async fn send_input_accepts_structured_items() {
    let (mut session, turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();
    let config = turn.config.as_ref().clone();
    let thread = manager
        .start_thread(config.clone())
        .await
        .expect("start thread");
    let agent_id = thread.thread_id;
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "send_input",
        function_payload(json!({
            "target": agent_id.to_string(),
            "items": [
                {"type": "mention", "name": "drive", "path": "app://google_drive"},
                {"type": "text", "text": "read the folder"}
            ]
        })),
    );
    SendInputHandler
        .handle(invocation)
        .await
        .expect("send_input should succeed");

    let expected = Op::UserInput {
        environments: None,
        items: vec![
            UserInput::Mention {
                name: "drive".to_string(),
                path: "app://google_drive".to_string(),
            },
            UserInput::Text {
                text: "read the folder".to_string(),
                text_elements: Vec::new(),
            },
        ],
        final_output_json_schema: None,
        responsesapi_client_metadata: None,
        additional_context: Default::default(),
        thread_settings: Default::default(),
    };
    let captured = manager
        .captured_ops()
        .into_iter()
        .find(|(id, op)| *id == agent_id && *op == expected);
    assert_eq!(captured, Some((agent_id, expected)));

    let _ = thread
        .thread
        .submit(Op::Shutdown {})
        .await
        .expect("shutdown should submit");
}

#[tokio::test]
async fn resume_agent_rejects_invalid_id() {
    let (session, turn) = make_session_and_context().await;
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "resume_agent",
        function_payload(json!({"id": "not-a-uuid"})),
    );
    let Err(err) = ResumeAgentHandler.handle(invocation).await else {
        panic!("invalid id should be rejected");
    };
    let FunctionCallError::RespondToModel(msg) = err else {
        panic!("expected respond-to-model error");
    };
    assert!(msg.starts_with("invalid agent id not-a-uuid:"));
}

#[tokio::test]
async fn resume_agent_reports_missing_agent() {
    let (mut session, turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();
    let agent_id = ThreadId::new();
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "resume_agent",
        function_payload(json!({"id": agent_id.to_string()})),
    );
    let Err(err) = ResumeAgentHandler.handle(invocation).await else {
        panic!("missing agent should be reported");
    };
    assert_eq!(
        err,
        FunctionCallError::RespondToModel(format!("agent with id {agent_id} not found"))
    );
}

#[tokio::test]
async fn resume_agent_noops_for_active_agent() {
    let (mut session, turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();
    let config = turn.config.as_ref().clone();
    let thread = manager
        .start_thread(config.clone())
        .await
        .expect("start thread");
    let agent_id = thread.thread_id;
    let status_before = manager.agent_control().get_status(agent_id).await;
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "resume_agent",
        function_payload(json!({"id": agent_id.to_string()})),
    );

    let output = ResumeAgentHandler
        .handle(invocation)
        .await
        .expect("resume_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: resume_agent::ResumeAgentResult =
        serde_json::from_str(&content).expect("resume_agent result should be json");
    assert_eq!(result.status, status_before);
    assert_eq!(success, Some(true));

    let thread_ids = manager.list_thread_ids().await;
    assert_eq!(thread_ids, vec![agent_id]);

    let _ = thread
        .thread
        .submit(Op::Shutdown {})
        .await
        .expect("shutdown should submit");
}

#[tokio::test]
async fn resume_agent_restores_closed_agent_and_accepts_send_input() {
    let (mut session, turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();
    let config = turn.config.as_ref().clone();
    let thread = manager
        .resume_thread_with_history(
            config.clone(),
            InitialHistory::Forked(vec![RolloutItem::ResponseItem(ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "materialized".to_string(),
                }],
                phase: None,
            })]),
            AuthManager::from_auth_for_testing(CodexAuth::from_api_key("dummy")),
            /*parent_trace*/ None,
        )
        .await
        .expect("start thread");
    let agent_id = thread.thread_id;
    let _ = manager
        .agent_control()
        .shutdown_live_agent(agent_id)
        .await
        .expect("shutdown agent");
    assert_eq!(
        manager.agent_control().get_status(agent_id).await,
        AgentStatus::NotFound
    );
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    let resume_invocation = invocation(
        session.clone(),
        turn.clone(),
        "resume_agent",
        function_payload(json!({"id": agent_id.to_string()})),
    );
    let output = ResumeAgentHandler
        .handle(resume_invocation)
        .await
        .expect("resume_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: resume_agent::ResumeAgentResult =
        serde_json::from_str(&content).expect("resume_agent result should be json");
    assert_ne!(result.status, AgentStatus::NotFound);
    assert_eq!(success, Some(true));

    let send_invocation = invocation(
        session,
        turn,
        "send_input",
        function_payload(json!({"target": agent_id.to_string(), "message": "hello"})),
    );
    let output = SendInputHandler
        .handle(send_invocation)
        .await
        .expect("send_input should succeed after resume");
    let (content, success) = expect_text_output(output);
    let result: serde_json::Value =
        serde_json::from_str(&content).expect("send_input result should be json");
    let submission_id = result
        .get("submission_id")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    assert!(!submission_id.is_empty());
    assert_eq!(success, Some(true));

    let _ = manager
        .agent_control()
        .shutdown_live_agent(agent_id)
        .await
        .expect("shutdown resumed agent");
}

#[tokio::test]
async fn resume_agent_rejects_when_depth_limit_exceeded() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();

    let max_depth = turn.config.agent_max_depth;
    turn.session_source = SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
        parent_thread_id: session.thread_id,
        depth: max_depth,
        agent_path: None,
        agent_nickname: None,
        agent_role: None,
    });

    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "resume_agent",
        function_payload(json!({"id": ThreadId::new().to_string()})),
    );
    let Err(err) = ResumeAgentHandler.handle(invocation).await else {
        panic!("resume should fail when depth limit exceeded");
    };
    assert_eq!(
        err,
        FunctionCallError::RespondToModel(
            "Agent depth limit reached. Solve the task yourself.".to_string()
        )
    );
}

#[tokio::test]
async fn wait_agent_rejects_non_positive_timeout() {
    let (session, turn) = make_session_and_context().await;
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "wait_agent",
        function_payload(json!({
            "targets": [ThreadId::new().to_string()],
            "timeout_ms": 0
        })),
    );
    let Err(err) = WaitAgentHandler::default().handle(invocation).await else {
        panic!("non-positive timeout should be rejected");
    };
    assert_eq!(
        err,
        FunctionCallError::RespondToModel("timeout_ms must be greater than zero".to_string())
    );
}

#[tokio::test]
async fn wait_agent_rejects_invalid_target() {
    let (session, turn) = make_session_and_context().await;
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "wait_agent",
        function_payload(json!({"targets": ["invalid"]})),
    );
    let Err(err) = WaitAgentHandler::default().handle(invocation).await else {
        panic!("invalid id should be rejected");
    };
    let FunctionCallError::RespondToModel(msg) = err else {
        panic!("expected respond-to-model error");
    };
    assert!(msg.starts_with("invalid agent id invalid:"));
}

#[tokio::test]
async fn wait_agent_rejects_empty_targets() {
    let (session, turn) = make_session_and_context().await;
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "wait_agent",
        function_payload(json!({"targets": []})),
    );
    let Err(err) = WaitAgentHandler::default().handle(invocation).await else {
        panic!("empty ids should be rejected");
    };
    assert_eq!(
        err,
        FunctionCallError::RespondToModel("agent ids must be non-empty".to_string())
    );
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_accepts_timeout_only_argument() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "boot worker",
                "task_name": "worker"
            })),
        ))
        .await
        .expect("spawn worker");
    let agent_id = session
        .services
        .agent_control
        .resolve_agent_reference(session.thread_id, &turn.session_source, "worker")
        .await
        .expect("worker should resolve");
    let worker_path = session
        .services
        .agent_control
        .get_agent_metadata(agent_id)
        .expect("worker metadata")
        .agent_path
        .expect("worker path");

    let wait_task = tokio::spawn({
        let session = session.clone();
        let turn = turn.clone();
        async move {
            WaitAgentHandlerV2::default()
                .handle(invocation(
                    session,
                    turn,
                    "wait_agent",
                    function_payload(json!({"timeout_ms": 10_000})),
                ))
                .await
        }
    });
    tokio::task::yield_now().await;

    session
        .input_queue
        .enqueue_mailbox_communication(InterAgentCommunication::new(
            worker_path,
            AgentPath::root(),
            Vec::new(),
            "hello from worker".to_string(),
            /*trigger_turn*/ false,
        ))
        .await;

    let output = wait_task
        .await
        .expect("wait task should join")
        .expect("timeout-only args should be accepted in v2 mode");
    let (content, success) = expect_text_output(output);
    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    assert_eq!(
        result,
        crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult {
            message: "Wait completed.".to_string(),
            timed_out: false,
            statuses: None,
            agent_statuses: None,
            empty_completions: vec![],
            wait_again_allowed: None,
        }
    );
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_rejects_timeout_below_configured_min() {
    let (session, mut turn) = make_session_and_context().await;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    config.multi_agent_v2.min_wait_timeout_ms = 50;
    config.multi_agent_v2.max_wait_timeout_ms = 1_000;
    config.multi_agent_v2.default_wait_timeout_ms = 50;
    set_turn_config(&mut turn, config);

    let Err(err) = WaitAgentHandlerV2::default()
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "wait_agent",
            function_payload(json!({"timeout_ms": 1})),
        ))
        .await
    else {
        panic!("timeout below configured minimum should be rejected");
    };
    assert_eq!(
        err,
        FunctionCallError::RespondToModel("timeout_ms must be at least 50".to_string())
    );
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_accepts_explicit_timeout_at_configured_min() {
    let (session, mut turn) = make_session_and_context().await;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    config.multi_agent_v2.min_wait_timeout_ms = 1;
    config.multi_agent_v2.max_wait_timeout_ms = 1_000;
    config.multi_agent_v2.default_wait_timeout_ms = 50;
    set_turn_config(&mut turn, config);

    let output = WaitAgentHandlerV2::default()
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "wait_agent",
            function_payload(json!({"timeout_ms": 1})),
        ))
        .await
        .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    assert_eq!(
        result,
        crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult {
            message: "Wait timed out.".to_string(),
            timed_out: true,
            statuses: None,
            agent_statuses: None,
            empty_completions: vec![],
            wait_again_allowed: None,
        }
    );
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_uses_configured_default_timeout() {
    let (session, mut turn) = make_session_and_context().await;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    config.multi_agent_v2.min_wait_timeout_ms = 1;
    config.multi_agent_v2.max_wait_timeout_ms = 1_000;
    config.multi_agent_v2.default_wait_timeout_ms = 50;
    set_turn_config(&mut turn, config);
    use_bedrock_provider(&mut turn);
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    let early = timeout(
        Duration::from_millis(/*millis*/ 20),
        WaitAgentHandlerV2::default().handle(invocation(
            session.clone(),
            turn.clone(),
            "wait_agent",
            function_payload(json!({})),
        )),
    )
    .await;
    assert!(
        early.is_err(),
        "wait_agent should not return before the configured default timeout"
    );

    let output = timeout(
        Duration::from_secs(/*secs*/ 1),
        WaitAgentHandlerV2::default().handle(invocation(
            session,
            turn,
            "wait_agent",
            function_payload(json!({})),
        )),
    )
    .await
    .expect("configured default should be shorter than the test timeout")
    .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    assert_eq!(
        result,
        crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult {
            message: "Wait timed out.".to_string(),
            timed_out: true,
            statuses: None,
            agent_statuses: None,
            empty_completions: vec![],
            wait_again_allowed: None,
        }
    );
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_allows_zero_configured_timeout() {
    let (session, mut turn) = make_session_and_context().await;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    config.multi_agent_v2.min_wait_timeout_ms = 0;
    config.multi_agent_v2.max_wait_timeout_ms = 0;
    config.multi_agent_v2.default_wait_timeout_ms = 0;
    set_turn_config(&mut turn, config);
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    let output = timeout(
        Duration::from_secs(/*secs*/ 1),
        WaitAgentHandlerV2::default().handle(invocation(
            session,
            turn,
            "wait_agent",
            function_payload(json!({})),
        )),
    )
    .await
    .expect("zero timeout should complete immediately")
    .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    assert_eq!(
        result,
        crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult {
            message: "Wait timed out.".to_string(),
            timed_out: true,
            statuses: None,
            agent_statuses: None,
            empty_completions: vec![],
            wait_again_allowed: None,
        }
    );
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_rejects_timeout_above_configured_max() {
    let (session, mut turn) = make_session_and_context().await;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    config.multi_agent_v2.min_wait_timeout_ms = 1;
    config.multi_agent_v2.max_wait_timeout_ms = 50;
    config.multi_agent_v2.default_wait_timeout_ms = 1;
    set_turn_config(&mut turn, config);

    let Err(err) = WaitAgentHandlerV2::default()
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "wait_agent",
            function_payload(json!({"timeout_ms": 500})),
        ))
        .await
    else {
        panic!("timeout above configured maximum should be rejected");
    };
    assert_eq!(
        err,
        FunctionCallError::RespondToModel("timeout_ms must be at most 50".to_string())
    );
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_accepts_explicit_timeout_at_configured_max() {
    let (session, mut turn) = make_session_and_context().await;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    config.multi_agent_v2.min_wait_timeout_ms = 1;
    config.multi_agent_v2.max_wait_timeout_ms = 1;
    config.multi_agent_v2.default_wait_timeout_ms = 1;
    set_turn_config(&mut turn, config);

    let output = WaitAgentHandlerV2::default()
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "wait_agent",
            function_payload(json!({"timeout_ms": 1})),
        ))
        .await
        .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    assert_eq!(
        result,
        crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult {
            message: "Wait timed out.".to_string(),
            timed_out: true,
            statuses: None,
            agent_statuses: None,
            empty_completions: vec![],
            wait_again_allowed: None,
        }
    );
    assert_eq!(success, None);
}

#[tokio::test]
async fn wait_agent_returns_not_found_for_missing_agents() {
    let (mut session, turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();
    let id_a = ThreadId::new();
    let id_b = ThreadId::new();
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "wait_agent",
        function_payload(json!({
            "targets": [id_a.to_string(), id_b.to_string()],
            "timeout_ms": 10_000
        })),
    );
    let output = WaitAgentHandler::default()
        .handle(invocation)
        .await
        .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    assert_eq!(
        result,
        wait::WaitAgentResult {
            status: HashMap::from([
                (id_a.to_string(), AgentStatus::NotFound),
                (id_b.to_string(), AgentStatus::NotFound),
            ]),
            timed_out: false
        }
    );
    assert_eq!(success, None);
}

#[tokio::test]
async fn wait_agent_times_out_when_status_is_not_final() {
    let (mut session, turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();
    let config = turn.config.as_ref().clone();
    let thread = manager
        .start_thread(config.clone())
        .await
        .expect("start thread");
    let agent_id = thread.thread_id;
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "wait_agent",
        function_payload(json!({
            "targets": [agent_id.to_string()],
            "timeout_ms": MIN_WAIT_TIMEOUT_MS
        })),
    );
    let output = WaitAgentHandler::default()
        .handle(invocation)
        .await
        .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    assert_eq!(
        result,
        wait::WaitAgentResult {
            status: HashMap::new(),
            timed_out: true
        }
    );
    assert_eq!(success, None);

    let _ = thread
        .thread
        .submit(Op::Shutdown {})
        .await
        .expect("shutdown should submit");
}

#[tokio::test]
async fn wait_agent_clamps_short_timeouts_to_minimum() {
    let (mut session, turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();
    let config = turn.config.as_ref().clone();
    let thread = manager
        .start_thread(config.clone())
        .await
        .expect("start thread");
    let agent_id = thread.thread_id;
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "wait_agent",
        function_payload(json!({
            "targets": [agent_id.to_string()],
            "timeout_ms": 10
        })),
    );

    let early = timeout(
        Duration::from_millis(50),
        WaitAgentHandler::default().handle(invocation),
    )
    .await;
    assert!(
        early.is_err(),
        "wait_agent should not return before the minimum timeout clamp"
    );

    let _ = thread
        .thread
        .submit(Op::Shutdown {})
        .await
        .expect("shutdown should submit");
}

#[tokio::test]
async fn wait_agent_returns_final_status_without_timeout() {
    let (mut session, turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();
    let config = turn.config.as_ref().clone();
    let thread = manager
        .start_thread(config.clone())
        .await
        .expect("start thread");
    let agent_id = thread.thread_id;
    let mut status_rx = manager
        .agent_control()
        .subscribe_status(agent_id)
        .await
        .expect("subscribe should succeed");

    let _ = thread
        .thread
        .submit(Op::Shutdown {})
        .await
        .expect("shutdown should submit");
    let _ = timeout(Duration::from_secs(1), status_rx.changed())
        .await
        .expect("shutdown status should arrive");

    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "wait_agent",
        function_payload(json!({
            "targets": [agent_id.to_string()],
            "timeout_ms": 10_000
        })),
    );
    let output = WaitAgentHandler::default()
        .handle(invocation)
        .await
        .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    assert_eq!(
        result,
        wait::WaitAgentResult {
            status: HashMap::from([(agent_id.to_string(), AgentStatus::Shutdown)]),
            timed_out: false
        }
    );
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_returns_summary_for_mailbox_activity() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);

    let session = Arc::new(session);
    let turn = Arc::new(turn);
    let spawn_output = SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "task_name": "test_process"
            })),
        ))
        .await
        .expect("spawn_agent should succeed");
    let _ = expect_text_output(spawn_output);

    let agent_id = session
        .services
        .agent_control
        .resolve_agent_reference(session.thread_id, &turn.session_source, "test_process")
        .await
        .expect("relative path should resolve");
    let worker_path = session
        .services
        .agent_control
        .get_agent_metadata(agent_id)
        .expect("worker metadata")
        .agent_path
        .expect("worker path");
    let wait_task = tokio::spawn({
        let session = session.clone();
        let turn = turn.clone();
        async move {
            WaitAgentHandlerV2::default()
                .handle(invocation(
                    session,
                    turn,
                    "wait_agent",
                    function_payload(json!({"timeout_ms": 10_000})),
                ))
                .await
        }
    });
    tokio::task::yield_now().await;

    session
        .input_queue
        .enqueue_mailbox_communication(InterAgentCommunication::new(
            worker_path,
            AgentPath::root(),
            Vec::new(),
            "completed".to_string(),
            /*trigger_turn*/ false,
        ))
        .await;

    let wait_output = wait_task
        .await
        .expect("wait task should join")
        .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(wait_output);
    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    assert_eq!(
        result,
        crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult {
            message: "Wait completed.".to_string(),
            timed_out: false,
            statuses: None,
            agent_statuses: None,
            empty_completions: vec![],
            wait_again_allowed: None,
        }
    );
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_returns_for_already_queued_mail() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "boot worker",
                "task_name": "worker"
            })),
        ))
        .await
        .expect("spawn worker");
    let agent_id = session
        .services
        .agent_control
        .resolve_agent_reference(session.thread_id, &turn.session_source, "worker")
        .await
        .expect("worker should resolve");
    let worker_path = session
        .services
        .agent_control
        .get_agent_metadata(agent_id)
        .expect("worker metadata")
        .agent_path
        .expect("worker path");

    session
        .input_queue
        .enqueue_mailbox_communication(InterAgentCommunication::new(
            worker_path,
            AgentPath::root(),
            Vec::new(),
            "already queued".to_string(),
            /*trigger_turn*/ false,
        ))
        .await;

    let output = timeout(
        Duration::from_millis(500),
        WaitAgentHandlerV2::default().handle(invocation(
            session,
            turn,
            "wait_agent",
            function_payload(json!({"timeout_ms": 10_000})),
        )),
    )
    .await
    .expect("already queued mail should complete wait_agent immediately")
    .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    // Non-Gemini path stays byte-identical: the Change 1 structured fields are skipped entirely
    // (this complements the timed-out raw-JSON assertion in the non-Gemini ignores-completed test).
    assert_eq!(
        content,
        r#"{"message":"Wait completed.","timed_out":false}"#
    );
    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    assert_eq!(
        result,
        crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult {
            message: "Wait completed.".to_string(),
            timed_out: false,
            statuses: None,
            agent_statuses: None,
            empty_completions: vec![],
            wait_again_allowed: None,
        }
    );
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_wakes_on_any_mailbox_notification() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    for task_name in ["worker_a", "worker_b"] {
        SpawnAgentHandlerV2::default()
            .handle(invocation(
                session.clone(),
                turn.clone(),
                "spawn_agent",
                function_payload(json!({
                    "message": format!("boot {task_name}"),
                    "task_name": task_name
                })),
            ))
            .await
            .expect("spawn worker");
    }
    let worker_b_id = session
        .services
        .agent_control
        .resolve_agent_reference(session.thread_id, &turn.session_source, "worker_b")
        .await
        .expect("worker_b should resolve");
    let worker_b_path = session
        .services
        .agent_control
        .get_agent_metadata(worker_b_id)
        .expect("worker_b metadata")
        .agent_path
        .expect("worker_b path");

    let wait_task = tokio::spawn({
        let session = session.clone();
        let turn = turn.clone();
        async move {
            WaitAgentHandlerV2::default()
                .handle(invocation(
                    session,
                    turn,
                    "wait_agent",
                    function_payload(json!({"timeout_ms": 10_000})),
                ))
                .await
        }
    });
    tokio::task::yield_now().await;

    session
        .input_queue
        .enqueue_mailbox_communication(InterAgentCommunication::new(
            worker_b_path,
            AgentPath::root(),
            Vec::new(),
            "from worker b".to_string(),
            /*trigger_turn*/ false,
        ))
        .await;

    let output = wait_task
        .await
        .expect("wait task should join")
        .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    assert_eq!(
        result,
        crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult {
            message: "Wait completed.".to_string(),
            timed_out: false,
            statuses: None,
            agent_statuses: None,
            empty_completions: vec![],
            wait_again_allowed: None,
        }
    );
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_does_not_return_completed_content() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "boot worker",
                "task_name": "worker"
            })),
        ))
        .await
        .expect("spawn worker");
    let agent_id = session
        .services
        .agent_control
        .resolve_agent_reference(session.thread_id, &turn.session_source, "worker")
        .await
        .expect("worker should resolve");
    let worker_path = session
        .services
        .agent_control
        .get_agent_metadata(agent_id)
        .expect("worker metadata")
        .agent_path
        .expect("worker path");
    let wait_task = tokio::spawn({
        let session = session.clone();
        let turn = turn.clone();
        async move {
            WaitAgentHandlerV2::default()
                .handle(invocation(
                    session,
                    turn,
                    "wait_agent",
                    function_payload(json!({"timeout_ms": 10_000})),
                ))
                .await
        }
    });
    tokio::task::yield_now().await;

    session
        .input_queue
        .enqueue_mailbox_communication(InterAgentCommunication::new(
            worker_path,
            AgentPath::root(),
            Vec::new(),
            "sensitive child output".to_string(),
            /*trigger_turn*/ false,
        ))
        .await;

    let output = wait_task
        .await
        .expect("wait task should join")
        .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    assert_eq!(
        result,
        crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult {
            message: "Wait completed.".to_string(),
            timed_out: false,
            statuses: None,
            agent_statuses: None,
            empty_completions: vec![],
            wait_again_allowed: None,
        }
    );
    assert!(!content.contains("sensitive child output"));
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_gemini_keeps_plain_message_for_bodied_completion() {
    let (session, turn, rx, manager) = gemini_wait_setup().await;
    let (agent_id, agent_nickname, agent_role, _path) =
        spawn_worker(&session, &turn, "worker").await;
    complete_worker_turn(&manager, agent_id, "child result").await;

    let output = timeout(
        Duration::from_millis(500),
        WaitAgentHandlerV2::new_with_output_mode(
            WaitAgentTimeoutOptions::default(),
            WaitAgentV2OutputMode::GeminiStatuses,
        )
        .handle(invocation(
            session.clone(),
            turn,
            "wait_agent",
            function_payload(json!({"timeout_ms": 10_000})),
        )),
    )
    .await
    .expect("completed child status should unblock wait_agent")
    .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    let completed = AgentStatus::Completed(Some("child result".to_string()));
    let expected_statuses = HashMap::from([(agent_id, completed.clone())]);
    let expected_agent_statuses = vec![CollabAgentStatusEntry {
        thread_id: agent_id,
        agent_nickname,
        agent_role,
        status: completed,
    }];
    assert_eq!(
        result,
        crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult {
            message: "Wait completed.".to_string(),
            timed_out: false,
            // Gemini full delivery drops the redundant `statuses` copy; the child body now
            // reaches the model only via `agent_statuses` (asserted below).
            statuses: None,
            agent_statuses: Some(expected_agent_statuses.clone()),
            empty_completions: vec![],
            wait_again_allowed: Some(true),
        }
    );
    let end_event = last_waiting_end_event(&rx).await;
    assert_eq!(
        (
            end_event.sender_thread_id,
            end_event.call_id,
            end_event.agent_statuses,
            end_event.statuses,
        ),
        (
            session.thread_id,
            "call-1".to_string(),
            expected_agent_statuses,
            expected_statuses,
        )
    );
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_gemini_full_delivery_omits_statuses_keeps_agent_statuses() {
    const MARKER: &str = "FULL_DELIVERY_REPORT_BODY_MARKER";
    let (session, turn, _rx, manager) = gemini_wait_setup().await;
    let (agent_id, agent_nickname, agent_role, _path) =
        spawn_worker(&session, &turn, "worker").await;
    complete_worker_turn(&manager, agent_id, MARKER).await;

    let output = timeout(
        Duration::from_millis(500),
        WaitAgentHandlerV2::new_with_output_mode(
            WaitAgentTimeoutOptions::default(),
            WaitAgentV2OutputMode::GeminiStatuses,
        )
        .handle(invocation(
            session,
            turn,
            "wait_agent",
            function_payload(json!({"timeout_ms": 10_000})),
        )),
    )
    .await
    .expect("completed child status should unblock wait_agent")
    .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);

    // Byte-level: the redundant `statuses` field is gone, while `agent_statuses` and the report
    // body are present. `"statuses":` (quote-prefixed) cannot appear inside `"agent_statuses":`
    // because the byte before `statuses` there is `_`, not `"`, so this check is unambiguous.
    assert!(
        !content.contains(r#""statuses":"#),
        "Gemini full delivery must not serialize a `statuses` field: {content}"
    );
    assert!(
        content.contains(r#""agent_statuses":"#),
        "Gemini full delivery must serialize `agent_statuses`: {content}"
    );
    assert!(
        content.contains(MARKER),
        "the child report body must reach the model via agent_statuses: {content}"
    );

    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    assert!(result.statuses.is_none());
    assert_eq!(
        result.agent_statuses,
        Some(vec![CollabAgentStatusEntry {
            thread_id: agent_id,
            agent_nickname,
            agent_role,
            status: AgentStatus::Completed(Some(MARKER.to_string())),
        }])
    );
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_gemini_directs_parent_after_empty_completion() {
    let (session, turn, _rx, manager) = gemini_wait_setup().await;
    let (agent_id, agent_nickname, agent_role, _path) =
        spawn_worker(&session, &turn, "worker").await;
    complete_worker_turn_without_report(&manager, agent_id).await;

    let output = timeout(
        Duration::from_millis(500),
        WaitAgentHandlerV2::new_with_output_mode(
            WaitAgentTimeoutOptions::default(),
            WaitAgentV2OutputMode::GeminiStatuses,
        )
        .handle(invocation(
            session,
            turn,
            "wait_agent",
            function_payload(json!({"timeout_ms": 10_000})),
        )),
    )
    .await
    .expect("empty completion should unblock wait_agent")
    .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    let display_label = agent_nickname
        .clone()
        .unwrap_or_else(|| agent_id.to_string());
    assert!(result.message.starts_with("Wait completed."));
    assert!(result.message.contains(&format!(
        "1 of 1 agents are terminal with no report: {display_label}."
    )));
    assert!(result.message.contains(
        "They are not running — calling wait_agent again returns immediately and cannot produce a report for them. To get a report, send the agent a single concrete follow-up first, then wait again; otherwise proceed with the results you already have. Do not re-spawn these agents."
    ));
    let message = result.message.clone();
    assert_eq!(
        result,
        crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult {
            message,
            timed_out: false,
            statuses: None,
            agent_statuses: Some(vec![CollabAgentStatusEntry {
                thread_id: agent_id,
                agent_nickname,
                agent_role,
                status: AgentStatus::Completed(None),
            }]),
            empty_completions: vec![display_label],
            wait_again_allowed: Some(false),
        }
    );
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_gemini_names_only_empty_completion_in_mixed_result() {
    let (session, turn, _rx, manager) = gemini_wait_setup().await;
    let (bodied_id, bodied_nickname, bodied_role, _bodied_path) =
        spawn_worker(&session, &turn, "bodied_worker").await;
    let (empty_id, empty_nickname, empty_role, _empty_path) =
        spawn_worker(&session, &turn, "empty_worker").await;
    complete_worker_turn(&manager, bodied_id, "bodied result").await;
    complete_worker_turn_without_report(&manager, empty_id).await;

    let output = timeout(
        Duration::from_millis(500),
        WaitAgentHandlerV2::new_with_output_mode(
            WaitAgentTimeoutOptions::default(),
            WaitAgentV2OutputMode::GeminiStatuses,
        )
        .handle(invocation(
            session,
            turn,
            "wait_agent",
            function_payload(json!({"timeout_ms": 10_000})),
        )),
    )
    .await
    .expect("mixed completions should unblock wait_agent")
    .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    let mut result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    let empty_label = empty_nickname
        .clone()
        .unwrap_or_else(|| empty_id.to_string());
    let expected_message = format!(
        "Wait completed. 1 of 2 agents are terminal with no report: {empty_label}. They are not running — calling wait_agent again returns immediately and cannot produce a report for them. To get a report, send the agent a single concrete follow-up first, then wait again; otherwise proceed with the results you already have. Do not re-spawn these agents."
    );
    let mut agent_statuses = result
        .agent_statuses
        .take()
        .expect("agent statuses should be present");
    agent_statuses.sort_by_key(|entry| entry.thread_id.to_string());
    let mut expected_agent_statuses = vec![
        CollabAgentStatusEntry {
            thread_id: bodied_id,
            agent_nickname: bodied_nickname,
            agent_role: bodied_role,
            status: AgentStatus::Completed(Some("bodied result".to_string())),
        },
        CollabAgentStatusEntry {
            thread_id: empty_id,
            agent_nickname: empty_nickname,
            agent_role: empty_role,
            status: AgentStatus::Completed(None),
        },
    ];
    expected_agent_statuses.sort_by_key(|entry| entry.thread_id.to_string());
    result.agent_statuses = Some(agent_statuses);
    assert_eq!(
        result,
        crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult {
            message: expected_message,
            timed_out: false,
            statuses: None,
            agent_statuses: Some(expected_agent_statuses),
            empty_completions: vec![empty_label],
            wait_again_allowed: Some(false),
        }
    );
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_gemini_empty_completion_sets_structured_signal() {
    let (session, turn, _rx, manager) = gemini_wait_setup().await;
    let (agent_id, agent_nickname, agent_role, _path) =
        spawn_worker(&session, &turn, "worker").await;
    complete_worker_turn_without_report(&manager, agent_id).await;
    let display_label = agent_nickname
        .clone()
        .unwrap_or_else(|| agent_id.to_string());

    let output = timeout(
        Duration::from_millis(500),
        WaitAgentHandlerV2::new_with_output_mode(
            WaitAgentTimeoutOptions::default(),
            WaitAgentV2OutputMode::GeminiStatuses,
        )
        .handle(invocation(
            session,
            turn,
            "wait_agent",
            function_payload(json!({"timeout_ms": 10_000})),
        )),
    )
    .await
    .expect("empty completion should unblock wait_agent")
    .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");

    // Machine-readable "do not re-wait" signal: the empty-completion agent is named and
    // wait_again_allowed is false (re-waiting would return immediately with no new report).
    assert_eq!(result.empty_completions, vec![display_label]);
    assert_eq!(result.wait_again_allowed, Some(false));
    // Gemini full delivery drops the redundant `statuses` copy; the child status reaches the
    // model only via `agent_statuses`.
    assert_eq!(result.statuses, None);
    assert_eq!(
        result.agent_statuses,
        Some(vec![CollabAgentStatusEntry {
            thread_id: agent_id,
            agent_nickname,
            agent_role,
            status: AgentStatus::Completed(None),
        }])
    );
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_gemini_redundant_repeat_returns_compact() {
    let (session, turn, _rx, manager) = gemini_wait_setup().await;
    let (bodied_id, _bodied_nickname, _bodied_role, _bodied_path) =
        spawn_worker(&session, &turn, "bodied_worker").await;
    let (empty_id, empty_nickname, _empty_role, _empty_path) =
        spawn_worker(&session, &turn, "empty_worker").await;
    complete_worker_turn(&manager, bodied_id, "BODIED_REPORT_BODY_MARKER").await;
    complete_worker_turn_without_report(&manager, empty_id).await;
    let empty_label = empty_nickname.unwrap_or_else(|| empty_id.to_string());

    // First wait: full delivery — carries the bodied report in full.
    let first = timeout(
        Duration::from_millis(500),
        WaitAgentHandlerV2::new_with_output_mode(
            WaitAgentTimeoutOptions::default(),
            WaitAgentV2OutputMode::GeminiStatuses,
        )
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "wait_agent",
            function_payload(json!({"timeout_ms": 10_000})),
        )),
    )
    .await
    .expect("first wait should return")
    .expect("wait_agent should succeed");
    let (first_content, _) = expect_text_output(first);
    assert!(
        first_content.contains("BODIED_REPORT_BODY_MARKER"),
        "first wait must deliver the bodied report in full: {first_content}"
    );
    let first_result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&first_content).expect("wait_agent result should be json");
    // Full delivery drops the redundant `statuses` copy; the bodied report reaches the model via
    // `agent_statuses` (proven by the BODIED_REPORT_BODY_MARKER content assert above).
    assert!(first_result.statuses.is_none());
    assert!(first_result.agent_statuses.is_some());
    assert_eq!(first_result.empty_completions, vec![empty_label.clone()]);
    assert_eq!(first_result.wait_again_allowed, Some(false));

    // Second wait over the same unchanged terminal set: compact — the already-delivered report
    // body is NOT re-injected, but the directive and structured fields ARE present.
    let second = timeout(
        Duration::from_millis(500),
        WaitAgentHandlerV2::new_with_output_mode(
            WaitAgentTimeoutOptions::default(),
            WaitAgentV2OutputMode::GeminiStatuses,
        )
        .handle(invocation(
            session,
            turn,
            "wait_agent",
            function_payload(json!({"timeout_ms": 10_000})),
        )),
    )
    .await
    .expect("second wait should return")
    .expect("wait_agent should succeed");
    let (second_content, second_success) = expect_text_output(second);
    assert!(
        !second_content.contains("BODIED_REPORT_BODY_MARKER"),
        "redundant repeat must not re-inject the already-delivered report body: {second_content}"
    );
    let second_result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&second_content).expect("wait_agent result should be json");
    assert!(second_result.message.starts_with("Wait completed."));
    assert!(second_result.message.contains("terminal with no report"));
    assert_eq!(second_result.statuses, None);
    assert_eq!(second_result.agent_statuses, None);
    assert_eq!(second_result.empty_completions, vec![empty_label]);
    assert_eq!(second_result.wait_again_allowed, Some(false));
    assert_eq!(second_success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_gemini_followup_reenables_full_delivery() {
    let (session, turn, _rx, manager) = gemini_wait_setup().await;
    let (agent_id, agent_nickname, agent_role, _path) =
        spawn_worker(&session, &turn, "worker").await;
    complete_worker_turn_without_report(&manager, agent_id).await;
    let display_label = agent_nickname
        .clone()
        .unwrap_or_else(|| agent_id.to_string());

    // First wait: empty completion → directive, wait_again_allowed false.
    let first = timeout(
        Duration::from_millis(500),
        WaitAgentHandlerV2::new_with_output_mode(
            WaitAgentTimeoutOptions::default(),
            WaitAgentV2OutputMode::GeminiStatuses,
        )
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "wait_agent",
            function_payload(json!({"timeout_ms": 10_000})),
        )),
    )
    .await
    .expect("first wait should return")
    .expect("wait_agent should succeed");
    let (first_content, _) = expect_text_output(first);
    let first_result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&first_content).expect("wait_agent result should be json");
    assert_eq!(first_result.empty_completions, vec![display_label]);
    assert_eq!(first_result.wait_again_allowed, Some(false));

    // A follow-up produces a report: the child leaves Completed(None) for Completed(Some(..)).
    complete_worker_turn(&manager, agent_id, "report after followup").await;

    // Next wait: the status changed, so this is a full delivery that carries the report once and
    // flips wait_again_allowed back to true with no outstanding empty completions.
    let second = timeout(
        Duration::from_millis(500),
        WaitAgentHandlerV2::new_with_output_mode(
            WaitAgentTimeoutOptions::default(),
            WaitAgentV2OutputMode::GeminiStatuses,
        )
        .handle(invocation(
            session,
            turn,
            "wait_agent",
            function_payload(json!({"timeout_ms": 10_000})),
        )),
    )
    .await
    .expect("second wait should return")
    .expect("wait_agent should succeed");
    let (second_content, _) = expect_text_output(second);
    assert!(
        second_content.contains("report after followup"),
        "post-follow-up wait must deliver the new report in full: {second_content}"
    );
    let second_result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&second_content).expect("wait_agent result should be json");
    // Full delivery drops the redundant `statuses` copy; the new report reaches the model via
    // `agent_statuses` only.
    assert_eq!(second_result.statuses, None);
    assert_eq!(
        second_result.agent_statuses,
        Some(vec![CollabAgentStatusEntry {
            thread_id: agent_id,
            agent_nickname,
            agent_role,
            status: AgentStatus::Completed(Some("report after followup".to_string())),
        }])
    );
    assert_eq!(second_result.empty_completions, Vec::<String>::new());
    assert_eq!(second_result.wait_again_allowed, Some(true));
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_non_gemini_ignores_completed_child_without_mailbox() {
    let (mut session, mut turn, rx) = make_session_and_context_with_rx().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    {
        let session = Arc::get_mut(&mut session).expect("session should be uniquely owned");
        session.services.agent_control = manager.agent_control();
        session.thread_id = root.thread_id;
    }
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    config.multi_agent_v2.min_wait_timeout_ms = 0;
    config.multi_agent_v2.max_wait_timeout_ms = 1_000;
    config.multi_agent_v2.default_wait_timeout_ms = 1;
    {
        let turn = Arc::get_mut(&mut turn).expect("turn should be uniquely owned");
        set_turn_config(turn, config);
        use_bedrock_provider(turn);
    }

    SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "boot worker",
                "task_name": "worker"
            })),
        ))
        .await
        .expect("spawn worker");
    let agent_id = session
        .services
        .agent_control
        .resolve_agent_reference(session.thread_id, &turn.session_source, "worker")
        .await
        .expect("worker should resolve");
    let child_thread = manager
        .get_thread(agent_id)
        .await
        .expect("worker thread should exist");
    let child_turn = child_thread.codex.session.new_default_turn().await;
    child_thread
        .codex
        .session
        .send_event_raw(Event {
            id: child_turn.sub_id.clone(),
            msg: EventMsg::TurnComplete(TurnCompleteEvent {
                turn_id: child_turn.sub_id.clone(),
                last_agent_message: Some("child result".to_string()),
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            }),
        })
        .await;

    let output = timeout(
        Duration::from_millis(500),
        WaitAgentHandlerV2::new_with_output_mode(
            WaitAgentTimeoutOptions::default(),
            WaitAgentV2OutputMode::GeminiStatuses,
        )
        .handle(invocation(
            session.clone(),
            turn,
            "wait_agent",
            function_payload(json!({"timeout_ms": 1})),
        )),
    )
    .await
    .expect("summary wait should return after its timeout")
    .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    assert_eq!(content, r#"{"message":"Wait timed out.","timed_out":true}"#);
    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    assert_eq!(
        result,
        crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult {
            message: "Wait timed out.".to_string(),
            timed_out: true,
            statuses: None,
            agent_statuses: None,
            empty_completions: vec![],
            wait_again_allowed: None,
        }
    );
    let end_event = last_waiting_end_event(&rx).await;
    assert_eq!(
        end_event.agent_statuses,
        Vec::<CollabAgentStatusEntry>::new()
    );
    assert_eq!(end_event.statuses, HashMap::new());
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_gemini_returns_immediately_without_live_children() {
    let (mut session, mut turn, rx) = make_session_and_context_with_rx().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    {
        let session = Arc::get_mut(&mut session).expect("session should be uniquely owned");
        session.services.agent_control = manager.agent_control();
        session.thread_id = root.thread_id;
    }
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    {
        let turn = Arc::get_mut(&mut turn).expect("turn should be uniquely owned");
        set_turn_config(turn, config);
        use_gemini_provider(turn);
    }

    let output = timeout(
        Duration::from_millis(500),
        WaitAgentHandlerV2::new_with_output_mode(
            WaitAgentTimeoutOptions::default(),
            WaitAgentV2OutputMode::GeminiStatuses,
        )
        .handle(invocation(
            session.clone(),
            turn,
            "wait_agent",
            function_payload(json!({"timeout_ms": 10_000})),
        )),
    )
    .await
    .expect("no live children should return immediately")
    .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    assert_eq!(
        result,
        crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult {
            message: "No live child agents to wait for.".to_string(),
            timed_out: false,
            statuses: Some(HashMap::new()),
            agent_statuses: Some(Vec::new()),
            empty_completions: vec![],
            wait_again_allowed: Some(true),
        }
    );
    let end_event = last_waiting_end_event(&rx).await;
    assert_eq!(
        end_event.agent_statuses,
        Vec::<CollabAgentStatusEntry>::new()
    );
    assert_eq!(end_event.statuses, HashMap::new());
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_missing_enumerated_child_is_terminal_not_found() {
    let (mut session, _turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();
    let missing_id = ThreadId::new();
    let receiver_agents = vec![CollabAgentRef {
        thread_id: missing_id,
        agent_nickname: Some("Missing".to_string()),
        agent_role: Some("explorer".to_string()),
    }];

    let subscriptions = crate::tools::handlers::multi_agents_v2::wait::subscribe_child_statuses(
        &session,
        &receiver_agents,
    )
    .await
    .expect("missing child should be represented as terminal NotFound");
    assert_eq!(
        subscriptions.terminal_statuses,
        HashMap::from([(missing_id, AgentStatus::NotFound)])
    );
    assert_eq!(subscriptions.status_rxs.len(), 0);
}

async fn gemini_wait_setup() -> (
    Arc<crate::session::session::Session>,
    Arc<TurnContext>,
    async_channel::Receiver<Event>,
    ThreadManager,
) {
    let (mut session, mut turn, rx) = make_session_and_context_with_rx().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    {
        let session = Arc::get_mut(&mut session).expect("session should be uniquely owned");
        session.services.agent_control = manager.agent_control();
        session.thread_id = root.thread_id;
    }
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    {
        let turn = Arc::get_mut(&mut turn).expect("turn should be uniquely owned");
        set_turn_config(turn, config);
        use_gemini_provider(turn);
    }
    (session, turn, rx, manager)
}

async fn spawn_worker(
    session: &Arc<crate::session::session::Session>,
    turn: &Arc<TurnContext>,
    task_name: &str,
) -> (ThreadId, Option<String>, Option<String>, AgentPath) {
    let parent_path = turn
        .session_source
        .get_agent_path()
        .unwrap_or_else(AgentPath::root);
    let worker_path = parent_path
        .join(task_name)
        .expect("worker path should join");
    // These tests drive child status transitions explicitly. Queue the initial
    // communication without starting a model turn so a missing provider credential
    // cannot race the status under test to `Errored`.
    let spawned_agent = Box::pin(session.services.agent_control.spawn_agent_with_metadata(
        (*turn.config).clone(),
        Op::InterAgentCommunication {
            communication: InterAgentCommunication::new(
                parent_path,
                worker_path.clone(),
                Vec::new(),
                format!("boot {task_name}"),
                /*trigger_turn*/ false,
            ),
        },
        Some(SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
            parent_thread_id: session.thread_id,
            depth: 1,
            agent_path: Some(worker_path),
            agent_nickname: None,
            agent_role: None,
        })),
        crate::agent::control::SpawnAgentOptions::default(),
    ))
    .await
    .expect("spawn worker should succeed");
    let agent_id = spawned_agent.thread_id;
    let metadata = spawned_agent.metadata;
    (
        agent_id,
        metadata.agent_nickname,
        metadata.agent_role,
        metadata.agent_path.expect("worker path"),
    )
}

async fn complete_worker_turn(manager: &ThreadManager, agent_id: ThreadId, message: &str) {
    complete_worker_turn_with_report(manager, agent_id, WorkerTurnReport::Present(message)).await;
}

async fn complete_worker_turn_without_report(manager: &ThreadManager, agent_id: ThreadId) {
    complete_worker_turn_with_report(manager, agent_id, WorkerTurnReport::Missing).await;
}

/// Drives a worker to a terminal `Errored` status (a determinate failure verdict),
/// used to pin O27's "an errored child is delivered, not chased" rule.
async fn error_worker_turn(manager: &ThreadManager, agent_id: ThreadId, message: &str) {
    let child_thread = manager
        .get_thread(agent_id)
        .await
        .expect("worker thread should exist");
    child_thread
        .codex
        .session
        .send_event_raw(Event {
            id: "worker-error".to_string(),
            msg: EventMsg::Error(ErrorEvent {
                message: message.to_string(),
                codex_error_info: None,
            }),
        })
        .await;
}

enum WorkerTurnReport<'a> {
    Present(&'a str),
    Missing,
}

async fn complete_worker_turn_with_report(
    manager: &ThreadManager,
    agent_id: ThreadId,
    report: WorkerTurnReport<'_>,
) {
    let child_thread = manager
        .get_thread(agent_id)
        .await
        .expect("worker thread should exist");
    let child_turn = child_thread.codex.session.new_default_turn().await;
    child_thread
        .codex
        .session
        .send_event_raw(Event {
            id: child_turn.sub_id.clone(),
            msg: EventMsg::TurnComplete(TurnCompleteEvent {
                turn_id: child_turn.sub_id.clone(),
                last_agent_message: match report {
                    WorkerTurnReport::Present(message) => Some(message.to_string()),
                    WorkerTurnReport::Missing => None,
                },
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            }),
        })
        .await;
}

struct ReleaseGatedParentTask {
    release: Arc<Notify>,
}

impl SessionTask for ReleaseGatedParentTask {
    fn kind(&self) -> TaskKind {
        TaskKind::Regular
    }

    fn span_name(&self) -> &'static str {
        "session_task.release_gated_parent"
    }

    async fn run(
        self: Arc<Self>,
        _session: Arc<SessionTaskContext>,
        _ctx: Arc<TurnContext>,
        _input: Vec<TurnInput>,
        cancellation_token: CancellationToken,
    ) -> Option<String> {
        tokio::select! {
            _ = self.release.notified() => None,
            _ = cancellation_token.cancelled() => None,
        }
    }
}

#[tokio::test]
async fn multi_agent_v2_gemini_spawned_child_captures_only_root_send_message_for_completion() {
    let (session, turn, _rx, manager) = gemini_wait_setup().await;
    let (worker_id, _worker_nickname, _worker_role, worker_path) =
        spawn_worker(&session, &turn, "worker").await;
    let (sibling_id, _sibling_nickname, _sibling_role, sibling_path) =
        spawn_worker(&session, &turn, "sibling").await;
    let worker_thread = manager
        .get_thread(worker_id)
        .await
        .expect("worker thread should exist");
    let worker_session = worker_thread.codex.session.clone();
    let worker_turn = worker_session.new_default_turn().await;
    assert_eq!(worker_turn.provider.info().wire_api, WireApi::GeminiNative);
    assert!(matches!(
        &worker_turn.session_source,
        SessionSource::SubAgent(SubAgentSource::ThreadSpawn { .. })
    ));

    let release = Arc::new(Notify::new());
    worker_session
        .spawn_task(
            worker_turn.clone(),
            Vec::new(),
            ReleaseGatedParentTask {
                release: Arc::clone(&release),
            },
        )
        .await;

    let sibling_output = SendMessageHandlerV2
        .handle(invocation(
            worker_session.clone(),
            worker_turn.clone(),
            "send_message",
            function_payload(json!({
                "target": sibling_path.as_str(),
                "message": "sibling report",
            })),
        ))
        .await
        .expect("send_message should accept sibling path");
    assert_eq!(
        expect_text_output(sibling_output),
        (String::new(), Some(true))
    );
    let turn_state = worker_session
        .input_queue
        .turn_state_for_sub_id(&worker_session.active_turn, &worker_turn.sub_id)
        .await
        .expect("worker turn state should exist");
    assert_eq!(
        turn_state
            .lock()
            .await
            .gemini_spawned_subagent_last_send_message_to_root
            .clone(),
        None
    );

    let root_output = SendMessageHandlerV2
        .handle(invocation(
            worker_session.clone(),
            worker_turn.clone(),
            "send_message",
            function_payload(json!({
                "target": "/root",
                "message": "root report",
            })),
        ))
        .await
        .expect("send_message should accept root path");
    assert_eq!(expect_text_output(root_output), (String::new(), Some(true)));
    assert_eq!(
        turn_state
            .lock()
            .await
            .gemini_spawned_subagent_last_send_message_to_root
            .clone(),
        Some("root report".to_string())
    );
    assert!(manager.captured_ops().iter().any(|(id, op)| {
        *id == sibling_id
            && matches!(
                op,
                Op::InterAgentCommunication { communication }
                    if communication.author == worker_path
                        && communication.recipient == sibling_path
                        && communication.content == "sibling report"
                        && !communication.trigger_turn
            )
    }));
    assert!(manager.captured_ops().iter().any(|(id, op)| {
        *id == session.thread_id
            && matches!(
                op,
                Op::InterAgentCommunication { communication }
                    if communication.author == worker_path
                        && communication.recipient == AgentPath::root()
                        && communication.content == "root report"
                        && !communication.trigger_turn
            )
    }));

    worker_session
        .abort_all_tasks(TurnAbortReason::Interrupted)
        .await;
}

async fn complete_worker_turn_and_notify_parent(
    manager: &ThreadManager,
    agent_id: ThreadId,
    message: &str,
) {
    let child_thread = manager
        .get_thread(agent_id)
        .await
        .expect("worker thread should exist");
    let child_turn = child_thread.codex.session.new_default_turn().await;
    child_thread
        .codex
        .session
        .send_event(
            child_turn.as_ref(),
            EventMsg::TurnComplete(TurnCompleteEvent {
                turn_id: child_turn.sub_id.clone(),
                last_agent_message: Some(message.to_string()),
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            }),
        )
        .await;
}

#[tokio::test]
async fn multi_agent_v2_gemini_busy_parent_completion_starts_synthesis_with_reports() {
    let (_seed_session, mut seed_turn) = make_session_and_context().await;
    let mut config = (*seed_turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut seed_turn, config);
    use_gemini_provider(&mut seed_turn);

    let manager = ThreadManager::with_models_provider_for_tests(
        CodexAuth::from_api_key("dummy"),
        seed_turn.config.model_provider.clone(),
    );
    let root = manager
        .start_thread((*seed_turn.config).clone())
        .await
        .expect("root thread should start");
    let parent_thread = root.thread;
    let session = parent_thread.codex.session.clone();
    let turn = session.new_default_turn().await;

    let (id_a, _nick_a, _role_a, path_a) = spawn_worker(&session, &turn, "worker_a").await;
    let (id_b, _nick_b, _role_b, path_b) = spawn_worker(&session, &turn, "worker_b").await;
    let expected_a = format_subagent_notification_message(
        path_a.as_str(),
        &AgentStatus::Completed(Some("a result".to_string())),
    );
    let expected_b = format_subagent_notification_message(
        path_b.as_str(),
        &AgentStatus::Completed(Some("b result".to_string())),
    );

    let release = Arc::new(Notify::new());
    session
        .spawn_task(
            turn.clone(),
            Vec::new(),
            ReleaseGatedParentTask {
                release: Arc::clone(&release),
            },
        )
        .await;
    assert!(
        session.active_turn.lock().await.is_some(),
        "parent turn should remain active before child completions"
    );

    complete_worker_turn_and_notify_parent(&manager, id_a, "a result").await;
    complete_worker_turn_and_notify_parent(&manager, id_b, "b result").await;

    timeout(Duration::from_secs(5), async {
        loop {
            let active_turn_present = session.active_turn.lock().await.is_some();
            if active_turn_present && session.input_queue.has_trigger_turn_mailbox_items().await {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("trigger-turn mailbox work should queue while the parent turn is active");

    release.notify_waiters();

    let followup_turn_id = match timeout(Duration::from_secs(5), async {
        loop {
            let event = parent_thread
                .next_event()
                .await
                .expect("parent event channel should stay open");
            if let EventMsg::TurnStarted(started) = event.msg
                && started.turn_id != turn.sub_id
            {
                break started.turn_id;
            }
        }
    })
    .await
    {
        Ok(turn_id) => turn_id,
        Err(_) => {
            panic!(
                "no follow-up turn: expected Gemini trigger-turn mailbox mail to start a synthesis turn after active parent completed"
            )
        }
    };
    assert_ne!(followup_turn_id, turn.sub_id);
    assert!(
        !session.input_queue.has_trigger_turn_mailbox_items().await,
        "follow-up turn should drain trigger-turn mailbox work"
    );

    if timeout(Duration::from_secs(5), async {
        loop {
            let history = session.clone_history().await;
            let items = history.raw_items();
            let notifications = items
                .iter()
                .filter(|item| SubagentNotification::matches_clean_response_item(item))
                .filter_map(|item| {
                    let ResponseItem::Message {
                        role,
                        content,
                        phase,
                        ..
                    } = item
                    else {
                        return None;
                    };
                    let [ContentItem::OutputText { text }] = content.as_slice() else {
                        return None;
                    };
                    (role == "assistant" && matches!(phase, Some(MessagePhase::Commentary)))
                        .then(|| text.clone())
                })
                .collect::<Vec<_>>();
            let serialized_notifications = items
                .iter()
                .filter_map(|item| match item {
                    ResponseItem::Message { content, .. } => {
                        InterAgentCommunication::from_message_content(content)
                    }
                    _ => None,
                })
                .filter(|communication| {
                    <SubagentNotification as crate::context::ContextualUserFragment>::matches_text(
                        &communication.content,
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(serialized_notifications, Vec::new());
            let has_expected_a = notifications.iter().any(|text| text == &expected_a);
            let has_expected_b = notifications.iter().any(|text| text == &expected_b);
            if has_expected_a && has_expected_b {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .is_err()
    {
        session.abort_all_tasks(TurnAbortReason::Interrupted).await;
        panic!(
            "follow-up turn lacks child report bodies: expected both child completion reports in parent-visible history"
        );
    }

    session.abort_all_tasks(TurnAbortReason::Interrupted).await;
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_gemini_blocks_until_all_children_terminal() {
    let (session, turn, _rx, manager) = gemini_wait_setup().await;

    let (id_a, nick_a, role_a, _path_a) = spawn_worker(&session, &turn, "worker_a").await;
    let (id_b, nick_b, role_b, _path_b) = spawn_worker(&session, &turn, "worker_b").await;

    // worker_a is already terminal at entry; worker_b is still running.
    complete_worker_turn(&manager, id_a, "a result").await;

    let wait_task = tokio::spawn({
        let session = session.clone();
        let turn = turn.clone();
        async move {
            WaitAgentHandlerV2::new_with_output_mode(
                WaitAgentTimeoutOptions::default(),
                WaitAgentV2OutputMode::GeminiStatuses,
            )
            .handle(invocation(
                session,
                turn,
                "wait_agent",
                function_payload(json!({"timeout_ms": 10_000})),
            ))
            .await
        }
    });

    // Give the wait time to subscribe and block; it must NOT return on the
    // already-terminal worker_a alone (this was the fan-out > 1 spin).
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(
        !wait_task.is_finished(),
        "wait_agent must keep blocking while worker_b is still running"
    );

    // Now finish worker_b -> the wait should resolve with BOTH statuses.
    complete_worker_turn(&manager, id_b, "b result").await;

    let output = timeout(Duration::from_secs(5), wait_task)
        .await
        .expect("wait_agent should resolve once all children are terminal")
        .expect("wait task should join")
        .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");

    assert_eq!(result.message, "Wait completed.".to_string());
    assert!(!result.timed_out);
    // Full delivery drops the redundant `statuses` copy; both bodies reach the model via
    // `agent_statuses` (asserted below).
    assert_eq!(result.statuses, None);

    let mut agent_statuses = result.agent_statuses.expect("agent statuses present");
    agent_statuses.sort_by_key(|entry| entry.thread_id.to_string());
    let mut expected_agent_statuses = vec![
        CollabAgentStatusEntry {
            thread_id: id_a,
            agent_nickname: nick_a,
            agent_role: role_a,
            status: AgentStatus::Completed(Some("a result".to_string())),
        },
        CollabAgentStatusEntry {
            thread_id: id_b,
            agent_nickname: nick_b,
            agent_role: role_b,
            status: AgentStatus::Completed(Some("b result".to_string())),
        },
    ];
    expected_agent_statuses.sort_by_key(|entry| entry.thread_id.to_string());
    assert_eq!(agent_statuses, expected_agent_statuses);
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_gemini_floors_short_explicit_timeout() {
    let (session, mut turn, rx, manager) = gemini_wait_setup().await;

    let (id_a, nick_a, role_a, _path_a) = spawn_worker(&session, &turn, "worker_a").await;
    let (id_b, nick_b, role_b, _path_b) = spawn_worker(&session, &turn, "worker_b").await;

    complete_worker_turn(&manager, id_a, "a result").await;

    {
        let mut config = (*turn.config).clone();
        config.multi_agent_v2.min_wait_timeout_ms = 0;
        config.multi_agent_v2.max_wait_timeout_ms = 30_000;
        config.multi_agent_v2.default_wait_timeout_ms = 50;
        let turn = Arc::get_mut(&mut turn).expect("turn should be uniquely owned");
        set_turn_config(turn, config);
    }

    let wait_task = tokio::spawn({
        let session = session.clone();
        let turn = turn.clone();
        async move {
            WaitAgentHandlerV2::new_with_output_mode(
                WaitAgentTimeoutOptions::default(),
                WaitAgentV2OutputMode::GeminiStatuses,
            )
            .handle(invocation(
                session,
                turn,
                "wait_agent",
                function_payload(json!({"timeout_ms": 50})),
            ))
            .await
        }
    });

    timeout(Duration::from_secs(2), async {
        loop {
            let event = rx.recv().await.expect("event channel should be open");
            if matches!(event.msg, EventMsg::CollabWaitingBegin(_)) {
                break;
            }
        }
    })
    .await
    .expect("wait_agent should emit CollabWaitingBegin");

    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(
        !wait_task.is_finished(),
        "Gemini status wait should floor an in-range short timeout"
    );

    complete_worker_turn_and_notify_parent(&manager, id_b, "b result").await;

    let output = timeout(Duration::from_secs(5), wait_task)
        .await
        .expect("wait_agent should resolve once all children are terminal")
        .expect("wait task should join")
        .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    let mut result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");

    let mut agent_statuses = result
        .agent_statuses
        .take()
        .expect("agent statuses present");
    agent_statuses.sort_by_key(|entry| entry.thread_id.to_string());
    let mut expected_agent_statuses = vec![
        CollabAgentStatusEntry {
            thread_id: id_a,
            agent_nickname: nick_a,
            agent_role: role_a,
            status: AgentStatus::Completed(Some("a result".to_string())),
        },
        CollabAgentStatusEntry {
            thread_id: id_b,
            agent_nickname: nick_b,
            agent_role: role_b,
            status: AgentStatus::Completed(Some("b result".to_string())),
        },
    ];
    expected_agent_statuses.sort_by_key(|entry| entry.thread_id.to_string());
    result.agent_statuses = Some(agent_statuses);
    assert_eq!(
        result,
        crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult {
            message: "Wait completed.".to_string(),
            timed_out: false,
            statuses: None,
            agent_statuses: Some(expected_agent_statuses),
            empty_completions: vec![],
            wait_again_allowed: Some(true),
        }
    );
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_gemini_ignores_spurious_mailbox_notification() {
    let (session, turn, _rx, manager) = gemini_wait_setup().await;

    let (id_a, nick_a, role_a, path_a) = spawn_worker(&session, &turn, "worker_a").await;
    let (id_b, nick_b, role_b, _path_b) = spawn_worker(&session, &turn, "worker_b").await;

    complete_worker_turn(&manager, id_a, "a result").await;

    let wait_task = tokio::spawn({
        let session = session.clone();
        let turn = turn.clone();
        async move {
            WaitAgentHandlerV2::new_with_output_mode(
                WaitAgentTimeoutOptions::default(),
                WaitAgentV2OutputMode::GeminiStatuses,
            )
            .handle(invocation(
                session,
                turn,
                "wait_agent",
                function_payload(json!({"timeout_ms": 10_000})),
            ))
            .await
        }
    });
    tokio::task::yield_now().await;

    // A child-completion notification (or any mailbox traffic) must NOT end the wait
    // while worker_b is still running -- this was the original spin source.
    for _ in 0..3 {
        session
            .input_queue
            .enqueue_mailbox_communication(InterAgentCommunication::new(
                path_a.clone(),
                AgentPath::root(),
                Vec::new(),
                "spurious".to_string(),
                /*trigger_turn*/ false,
            ))
            .await;
    }
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(
        !wait_task.is_finished(),
        "spurious mailbox notifications must not end the wait early"
    );

    complete_worker_turn(&manager, id_b, "b result").await;

    let output = timeout(Duration::from_secs(5), wait_task)
        .await
        .expect("wait_agent should resolve once all children are terminal")
        .expect("wait task should join")
        .expect("wait_agent should succeed");
    let (content, _success) = expect_text_output(output);
    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    assert!(!result.timed_out);
    assert_eq!(result.message, "Wait completed.".to_string());
    // Full delivery drops the redundant `statuses` copy; both bodies reach the model via
    // `agent_statuses` only.
    assert_eq!(result.statuses, None);
    let mut agent_statuses = result.agent_statuses.expect("agent statuses present");
    agent_statuses.sort_by_key(|entry| entry.thread_id.to_string());
    let mut expected_agent_statuses = vec![
        CollabAgentStatusEntry {
            thread_id: id_a,
            agent_nickname: nick_a,
            agent_role: role_a,
            status: AgentStatus::Completed(Some("a result".to_string())),
        },
        CollabAgentStatusEntry {
            thread_id: id_b,
            agent_nickname: nick_b,
            agent_role: role_b,
            status: AgentStatus::Completed(Some("b result".to_string())),
        },
    ];
    expected_agent_statuses.sort_by_key(|entry| entry.thread_id.to_string());
    assert_eq!(agent_statuses, expected_agent_statuses);
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_gemini_times_out_with_partial_statuses() {
    let (session, mut turn, _rx, manager) = gemini_wait_setup().await;

    let (id_a, nick_a, role_a, _path_a) = spawn_worker(&session, &turn, "worker_a").await;
    let (_id_b, _nick_b, _role_b, _path_b) = spawn_worker(&session, &turn, "worker_b").await;

    // worker_a completes; worker_b stays running so the deadline must fire.
    complete_worker_turn(&manager, id_a, "a result").await;

    // Shrink the wait bounds so the deadline elapses quickly.
    {
        let mut config = (*turn.config).clone();
        config.multi_agent_v2.min_wait_timeout_ms = 0;
        config.multi_agent_v2.max_wait_timeout_ms = 200;
        config.multi_agent_v2.default_wait_timeout_ms = 1;
        let turn = Arc::get_mut(&mut turn).expect("turn should be uniquely owned");
        set_turn_config(turn, config);
    }

    let output = timeout(
        Duration::from_secs(5),
        WaitAgentHandlerV2::new_with_output_mode(
            WaitAgentTimeoutOptions::default(),
            WaitAgentV2OutputMode::GeminiStatuses,
        )
        .handle(invocation(
            session.clone(),
            turn,
            "wait_agent",
            function_payload(json!({"timeout_ms": 200})),
        )),
    )
    .await
    .expect("wait_agent should time out within the test budget")
    .expect("wait_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult =
        serde_json::from_str(&content).expect("wait_agent result should be json");
    assert!(
        result.timed_out,
        "deadline should fire while worker_b is still running"
    );
    assert_eq!(result.message, "Wait timed out.".to_string());
    // Full delivery drops the redundant `statuses` copy; only the terminal child is reported on
    // timeout, and it reaches the model via `agent_statuses` only.
    assert_eq!(result.statuses, None);
    assert_eq!(
        result.agent_statuses,
        Some(vec![CollabAgentStatusEntry {
            thread_id: id_a,
            agent_nickname: nick_a,
            agent_role: role_a,
            status: AgentStatus::Completed(Some("a result".to_string())),
        }])
    );
    assert_eq!(success, None);
}

#[tokio::test]
async fn multi_agent_v2_close_agent_accepts_task_name_target() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);

    let session = Arc::new(session);
    let turn = Arc::new(turn);
    SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "task_name": "worker"
            })),
        ))
        .await
        .expect("spawn_agent should succeed");

    let agent_id = session
        .services
        .agent_control
        .resolve_agent_reference(session.thread_id, &turn.session_source, "worker")
        .await
        .expect("worker path should resolve");

    let output = CloseAgentHandlerV2
        .handle(invocation(
            session,
            turn,
            "close_agent",
            function_payload(json!({"target": "worker"})),
        ))
        .await
        .expect("close_agent should succeed for v2 task names");
    let (content, success) = expect_text_output(output);
    let result: close_agent::CloseAgentResult =
        serde_json::from_str(&content).expect("close_agent result should be json");
    assert_ne!(result.previous_status, AgentStatus::NotFound);
    assert_eq!(success, Some(true));
    assert_eq!(
        manager.agent_control().get_status(agent_id).await,
        AgentStatus::NotFound
    );
}

#[tokio::test]
async fn multi_agent_v2_close_agent_reaps_stale_task_name_target() {
    let (mut session, mut turn) = make_session_and_context().await;
    let mut config = (*turn.config).clone();
    config.multi_agent_v2.max_concurrent_threads_per_session = 2;
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    config
        .features
        .enable(Feature::Sqlite)
        .expect("test config should allow sqlite");
    let state_db = init_state_db(&config)
        .await
        .expect("sqlite state db should initialize");
    let manager = ThreadManager::with_models_provider_home_and_state_for_tests(
        CodexAuth::from_api_key("dummy"),
        config.model_provider.clone(),
        config.codex_home.to_path_buf(),
        Arc::new(codex_exec_server::EnvironmentManager::default_for_tests()),
        Some(state_db.clone()),
    );
    let root = manager
        .start_thread(config.clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    set_turn_config(&mut turn, config.clone());

    let session = Arc::new(session);
    let turn = Arc::new(turn);
    SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "task_name": "worker"
            })),
        ))
        .await
        .expect("spawn_agent should succeed");

    let agent_id = session
        .services
        .agent_control
        .resolve_agent_reference(session.thread_id, &turn.session_source, "worker")
        .await
        .expect("worker path should resolve");
    let stale_thread = manager
        .remove_thread(&agent_id)
        .await
        .expect("worker thread should be loaded before removal");
    stale_thread
        .submit(Op::Shutdown {})
        .await
        .expect("removed worker thread should still accept shutdown");
    stale_thread.wait_until_terminated().await;

    let output = CloseAgentHandlerV2
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "close_agent",
            function_payload(json!({"target": "worker"})),
        ))
        .await
        .expect("close_agent should reap stale v2 task names");
    let (content, success) = expect_text_output(output);
    let result: close_agent::CloseAgentResult =
        serde_json::from_str(&content).expect("close_agent result should be json");
    assert_eq!(result.previous_status, AgentStatus::NotFound);
    assert_eq!(success, Some(true));

    let open_children = state_db
        .list_thread_spawn_children_with_status(
            root.thread_id,
            DirectionalThreadSpawnEdgeStatus::Open,
        )
        .await
        .expect("open children should load");
    assert_eq!(open_children, Vec::<ThreadId>::new());
    let closed_children = state_db
        .list_thread_spawn_children_with_status(
            root.thread_id,
            DirectionalThreadSpawnEdgeStatus::Closed,
        )
        .await
        .expect("closed children should load");
    assert_eq!(closed_children, vec![agent_id]);

    SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo again",
                "task_name": "replacement"
            })),
        ))
        .await
        .expect("spawn_agent should succeed after stale close releases the slot");
    let replacement_id = session
        .services
        .agent_control
        .resolve_agent_reference(session.thread_id, &turn.session_source, "replacement")
        .await
        .expect("replacement path should resolve");
    let _ = session
        .services
        .agent_control
        .shutdown_live_agent(replacement_id)
        .await
        .expect("replacement should shut down");
}

#[tokio::test]
async fn multi_agent_v2_close_agent_rejects_root_target_and_id() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);

    let session = Arc::new(session);
    let turn = Arc::new(turn);
    let root_path_error = CloseAgentHandlerV2
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "close_agent",
            function_payload(json!({"target": "/root"})),
        ))
        .await
        .err()
        .expect("close_agent should reject the root path");
    assert_eq!(
        root_path_error,
        FunctionCallError::RespondToModel("root is not a spawned agent".to_string())
    );

    let root_id_error = CloseAgentHandlerV2
        .handle(invocation(
            session,
            turn,
            "close_agent",
            function_payload(json!({"target": root.thread_id.to_string()})),
        ))
        .await
        .err()
        .expect("close_agent should reject the root thread id");
    assert_eq!(
        root_id_error,
        FunctionCallError::RespondToModel("root is not a spawned agent".to_string())
    );
}

#[tokio::test]
async fn multi_agent_v2_close_agent_rejects_self_target_by_id() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;

    let child_path = AgentPath::try_from("/root/worker").expect("agent path");
    let child_thread_id = session
        .services
        .agent_control
        .spawn_agent_with_metadata(
            (*turn.config).clone(),
            vec![UserInput::Text {
                text: "inspect this repo".to_string(),
                text_elements: Vec::new(),
            }]
            .into(),
            Some(SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
                parent_thread_id: root.thread_id,
                depth: 1,
                agent_path: Some(child_path.clone()),
                agent_nickname: None,
                agent_role: None,
            })),
            crate::agent::control::SpawnAgentOptions::default(),
        )
        .await
        .expect("worker spawn should succeed")
        .thread_id;
    session.thread_id = child_thread_id;
    turn.session_source = SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
        parent_thread_id: root.thread_id,
        depth: 1,
        agent_path: Some(child_path),
        agent_nickname: None,
        agent_role: None,
    });

    let err = CloseAgentHandlerV2
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "close_agent",
            function_payload(json!({"target": child_thread_id.to_string()})),
        ))
        .await
        .err()
        .expect("close_agent should reject self-target by id");
    assert_eq!(
        err,
        FunctionCallError::RespondToModel(
            "an agent cannot close itself; return your result and let the parent close you if needed"
                .to_string()
        )
    );
}

#[tokio::test]
async fn multi_agent_v2_close_agent_rejects_self_target_by_task_name() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::MultiAgentV2)
        .expect("test config should allow feature update");
    set_turn_config(&mut turn, config);
    let root = manager
        .start_thread((*turn.config).clone())
        .await
        .expect("root thread should start");
    session.services.agent_control = manager.agent_control();
    session.thread_id = root.thread_id;

    let child_path = AgentPath::try_from("/root/worker").expect("agent path");
    let child_thread_id = session
        .services
        .agent_control
        .spawn_agent_with_metadata(
            (*turn.config).clone(),
            vec![UserInput::Text {
                text: "inspect this repo".to_string(),
                text_elements: Vec::new(),
            }]
            .into(),
            Some(SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
                parent_thread_id: root.thread_id,
                depth: 1,
                agent_path: Some(child_path.clone()),
                agent_nickname: None,
                agent_role: None,
            })),
            crate::agent::control::SpawnAgentOptions::default(),
        )
        .await
        .expect("worker spawn should succeed")
        .thread_id;
    session.thread_id = child_thread_id;
    turn.session_source = SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
        parent_thread_id: root.thread_id,
        depth: 1,
        agent_path: Some(child_path.clone()),
        agent_nickname: None,
        agent_role: None,
    });

    let err = CloseAgentHandlerV2
        .handle(invocation(
            Arc::new(session),
            Arc::new(turn),
            "close_agent",
            function_payload(json!({"target": child_path.to_string()})),
        ))
        .await
        .err()
        .expect("close_agent should reject self-target by task name");
    assert_eq!(
        err,
        FunctionCallError::RespondToModel(
            "an agent cannot close itself; return your result and let the parent close you if needed"
                .to_string()
        )
    );
}

#[tokio::test]
async fn close_agent_submits_shutdown_and_returns_previous_status() {
    let (mut session, turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();
    let config = turn.config.as_ref().clone();
    let thread = manager
        .start_thread(config.clone())
        .await
        .expect("start thread");
    let agent_id = thread.thread_id;
    let status_before = manager.agent_control().get_status(agent_id).await;

    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "close_agent",
        function_payload(json!({"target": agent_id.to_string()})),
    );
    let output = CloseAgentHandler
        .handle(invocation)
        .await
        .expect("close_agent should succeed");
    let (content, success) = expect_text_output(output);
    let result: close_agent::CloseAgentResult =
        serde_json::from_str(&content).expect("close_agent result should be json");
    assert_eq!(result.previous_status, status_before);
    assert_eq!(success, Some(true));

    let ops = manager.captured_ops();
    let submitted_shutdown = ops
        .iter()
        .any(|(id, op)| *id == agent_id && matches!(op, Op::Shutdown));
    assert_eq!(submitted_shutdown, true);

    let status_after = manager.agent_control().get_status(agent_id).await;
    assert_eq!(status_after, AgentStatus::NotFound);
}

#[tokio::test]
async fn tool_handlers_cascade_close_and_resume_and_keep_explicitly_closed_subtrees_closed() {
    let (_session, turn) = make_session_and_context().await;
    let mut config = turn.config.as_ref().clone();
    config.agent_max_depth = 3;
    config
        .features
        .enable(Feature::Sqlite)
        .expect("test config should allow sqlite");
    let state_db = init_state_db(&config).await;
    let manager = ThreadManager::new(
        &config,
        AuthManager::from_auth_for_testing(CodexAuth::from_api_key("dummy")),
        SessionSource::Exec,
        Arc::new(codex_exec_server::EnvironmentManager::default_for_tests()),
        empty_extension_registry(),
        /*analytics_events_client*/ None,
        thread_store_from_config(&config, state_db.clone()),
        state_db.clone(),
        "11111111-1111-4111-8111-111111111111".to_string(),
        /*attestation_provider*/ None,
    );

    let parent = manager
        .start_thread(config.clone())
        .await
        .expect("parent thread should start");
    let parent_thread_id = parent.thread_id;
    let parent_session = parent.thread.codex.session.clone();

    let child_turn = parent_session.new_default_turn().await;
    let child_spawn_output = SpawnAgentHandler::default()
        .handle(invocation(
            parent_session.clone(),
            child_turn,
            "spawn_agent",
            function_payload(json!({"message": "hello child"})),
        ))
        .await
        .expect("child spawn should succeed");
    let (child_content, child_success) = expect_text_output(child_spawn_output);
    let child_result: serde_json::Value =
        serde_json::from_str(&child_content).expect("child spawn result should be json");
    let child_thread_id = parse_agent_id(
        child_result
            .get("agent_id")
            .and_then(serde_json::Value::as_str)
            .expect("child spawn result should include agent_id"),
    );
    assert_eq!(child_success, Some(true));

    let child_thread = manager
        .get_thread(child_thread_id)
        .await
        .expect("child thread should exist");
    let child_session = child_thread.codex.session.clone();
    let grandchild_spawn_output = SpawnAgentHandler::default()
        .handle(invocation(
            child_session.clone(),
            child_session.new_default_turn().await,
            "spawn_agent",
            function_payload(json!({"message": "hello grandchild"})),
        ))
        .await
        .expect("grandchild spawn should succeed");
    let (grandchild_content, grandchild_success) = expect_text_output(grandchild_spawn_output);
    let grandchild_result: serde_json::Value =
        serde_json::from_str(&grandchild_content).expect("grandchild spawn result should be json");
    let grandchild_thread_id = parse_agent_id(
        grandchild_result
            .get("agent_id")
            .and_then(serde_json::Value::as_str)
            .expect("grandchild spawn result should include agent_id"),
    );
    assert_eq!(grandchild_success, Some(true));

    let close_output = CloseAgentHandler
        .handle(invocation(
            parent_session.clone(),
            parent_session.new_default_turn().await,
            "close_agent",
            function_payload(json!({"target": child_thread_id.to_string()})),
        ))
        .await
        .expect("close_agent should close the child subtree");
    let (close_content, close_success) = expect_text_output(close_output);
    let close_result: close_agent::CloseAgentResult =
        serde_json::from_str(&close_content).expect("close_agent result should be json");
    assert_ne!(close_result.previous_status, AgentStatus::NotFound);
    assert_eq!(close_success, Some(true));
    assert_eq!(
        manager.agent_control().get_status(child_thread_id).await,
        AgentStatus::NotFound
    );
    assert_eq!(
        manager
            .agent_control()
            .get_status(grandchild_thread_id)
            .await,
        AgentStatus::NotFound
    );

    let child_resume_output = ResumeAgentHandler
        .handle(invocation(
            parent_session.clone(),
            parent_session.new_default_turn().await,
            "resume_agent",
            function_payload(json!({"id": child_thread_id.to_string()})),
        ))
        .await
        .expect("resume_agent should reopen the child subtree");
    let (child_resume_content, child_resume_success) = expect_text_output(child_resume_output);
    let child_resume_result: resume_agent::ResumeAgentResult =
        serde_json::from_str(&child_resume_content).expect("resume result should be json");
    assert_ne!(child_resume_result.status, AgentStatus::NotFound);
    assert_eq!(child_resume_success, Some(true));
    assert_ne!(
        manager.agent_control().get_status(child_thread_id).await,
        AgentStatus::NotFound
    );
    assert_ne!(
        manager
            .agent_control()
            .get_status(grandchild_thread_id)
            .await,
        AgentStatus::NotFound
    );

    let close_again_output = CloseAgentHandler
        .handle(invocation(
            parent_session.clone(),
            parent_session.new_default_turn().await,
            "close_agent",
            function_payload(json!({"target": child_thread_id.to_string()})),
        ))
        .await
        .expect("close_agent should be repeatable for the child subtree");
    let (close_again_content, close_again_success) = expect_text_output(close_again_output);
    let close_again_result: close_agent::CloseAgentResult =
        serde_json::from_str(&close_again_content)
            .expect("second close_agent result should be json");
    assert_ne!(close_again_result.previous_status, AgentStatus::NotFound);
    assert_eq!(close_again_success, Some(true));
    assert_eq!(
        manager.agent_control().get_status(child_thread_id).await,
        AgentStatus::NotFound
    );
    assert_eq!(
        manager
            .agent_control()
            .get_status(grandchild_thread_id)
            .await,
        AgentStatus::NotFound
    );

    let operator = manager
        .start_thread(config.clone())
        .await
        .expect("operator thread should start");
    let operator_session = operator.thread.codex.session.clone();
    let _ = manager
        .agent_control()
        .shutdown_live_agent(parent_thread_id)
        .await
        .expect("parent shutdown should succeed");
    assert_eq!(
        manager.agent_control().get_status(parent_thread_id).await,
        AgentStatus::NotFound
    );

    let parent_resume_output = ResumeAgentHandler
        .handle(invocation(
            operator_session,
            operator.thread.codex.session.new_default_turn().await,
            "resume_agent",
            function_payload(json!({"id": parent_thread_id.to_string()})),
        ))
        .await
        .expect("resume_agent should reopen the parent thread");
    let (parent_resume_content, parent_resume_success) = expect_text_output(parent_resume_output);
    let parent_resume_result: resume_agent::ResumeAgentResult =
        serde_json::from_str(&parent_resume_content).expect("parent resume result should be json");
    assert_ne!(parent_resume_result.status, AgentStatus::NotFound);
    assert_eq!(parent_resume_success, Some(true));
    assert_ne!(
        manager.agent_control().get_status(parent_thread_id).await,
        AgentStatus::NotFound
    );
    assert_eq!(
        manager.agent_control().get_status(child_thread_id).await,
        AgentStatus::NotFound
    );
    assert_eq!(
        manager
            .agent_control()
            .get_status(grandchild_thread_id)
            .await,
        AgentStatus::NotFound
    );

    let shutdown_report = manager
        .shutdown_all_threads_bounded(Duration::from_secs(5))
        .await;
    assert_eq!(shutdown_report.submit_failed, Vec::<ThreadId>::new());
    assert_eq!(shutdown_report.timed_out, Vec::<ThreadId>::new());
}

#[tokio::test]
async fn build_agent_spawn_config_uses_turn_context_values() {
    fn pick_allowed_sandbox_policy(
        permissions: &crate::config::Permissions,
        base: SandboxPolicy,
        cwd: &std::path::Path,
    ) -> SandboxPolicy {
        let candidates = [
            SandboxPolicy::new_read_only_policy(),
            SandboxPolicy::new_workspace_write_policy(),
            SandboxPolicy::DangerFullAccess,
        ];
        candidates
            .into_iter()
            .find(|candidate| {
                if *candidate == base {
                    return false;
                }
                permissions
                    .can_set_legacy_sandbox_policy(candidate, cwd)
                    .is_ok()
            })
            .unwrap_or(base)
    }

    let (_session, mut turn) = make_session_and_context().await;
    let base_instructions = BaseInstructions {
        text: "base".to_string(),
    };
    turn.developer_instructions = Some("dev".to_string());
    turn.compact_prompt = Some("compact".to_string());
    turn.shell_environment_policy = ShellEnvironmentPolicy {
        use_profile: true,
        ..ShellEnvironmentPolicy::default()
    };
    let temp_dir = tempfile::tempdir().expect("temp dir");
    #[allow(deprecated)]
    {
        turn.cwd = temp_dir.abs();
    }
    turn.codex_linux_sandbox_exe = Some(PathBuf::from("/bin/echo"));
    #[allow(deprecated)]
    let turn_cwd = turn.cwd.clone();
    let sandbox_policy = pick_allowed_sandbox_policy(
        &turn.config.permissions,
        turn.config.legacy_sandbox_policy(),
        turn_cwd.as_path(),
    );
    let file_system_sandbox_policy =
        FileSystemSandboxPolicy::from_legacy_sandbox_policy_for_cwd(&sandbox_policy, &turn_cwd);
    let network_sandbox_policy = NetworkSandboxPolicy::from(&sandbox_policy);
    let permission_profile = PermissionProfile::from_runtime_permissions_with_enforcement(
        SandboxEnforcement::from_legacy_sandbox_policy(&sandbox_policy),
        &file_system_sandbox_policy,
        network_sandbox_policy,
    );
    turn.permission_profile = permission_profile.clone();
    turn.approval_policy
        .set(AskForApproval::OnRequest)
        .expect("approval policy set");

    let config = build_agent_spawn_config(&base_instructions, &turn).expect("spawn config");
    let mut expected = (*turn.config).clone();
    expected.base_instructions = Some(base_instructions.text);
    expected.model = Some(turn.model_info.slug.clone());
    expected.model_provider = turn.provider.info().clone();
    expected.model_reasoning_effort = turn.reasoning_effort;
    expected.model_reasoning_summary = Some(turn.reasoning_summary);
    expected.developer_instructions = turn.developer_instructions.clone();
    expected.compact_prompt = turn.compact_prompt.clone();
    expected.permissions.shell_environment_policy = turn.shell_environment_policy.clone();
    expected.codex_linux_sandbox_exe = turn.codex_linux_sandbox_exe.clone();
    #[allow(deprecated)]
    {
        expected.cwd = turn.cwd.clone();
    }
    expected
        .permissions
        .approval_policy
        .set(AskForApproval::OnRequest)
        .expect("approval policy set");
    expected
        .permissions
        .set_permission_profile(permission_profile)
        .expect("permission profile set");
    assert_eq!(config, expected);
}

#[tokio::test]
async fn build_agent_spawn_config_preserves_base_user_instructions() {
    let (_session, mut turn) = make_session_and_context().await;
    let mut base_config = (*turn.config).clone();
    base_config.user_instructions = Some("base-user".to_string());
    turn.user_instructions = Some("resolved-user".to_string());
    turn.config = Arc::new(base_config.clone());
    let base_instructions = BaseInstructions {
        text: "base".to_string(),
    };

    let config = build_agent_spawn_config(&base_instructions, &turn).expect("spawn config");

    assert_eq!(config.user_instructions, base_config.user_instructions);
}

#[tokio::test]
async fn build_agent_resume_config_clears_base_instructions() {
    let (_session, mut turn) = make_session_and_context().await;
    let mut base_config = (*turn.config).clone();
    base_config.base_instructions = Some("caller-base".to_string());
    turn.config = Arc::new(base_config);
    turn.approval_policy
        .set(AskForApproval::OnRequest)
        .expect("approval policy set");

    let config = build_agent_resume_config(&turn).expect("resume config");

    let mut expected = (*turn.config).clone();
    expected.base_instructions = None;
    expected.model = Some(turn.model_info.slug.clone());
    expected.model_provider = turn.provider.info().clone();
    expected.model_reasoning_effort = turn.reasoning_effort;
    expected.model_reasoning_summary = Some(turn.reasoning_summary);
    expected.developer_instructions = turn.developer_instructions.clone();
    expected.compact_prompt = turn.compact_prompt.clone();
    expected.permissions.shell_environment_policy = turn.shell_environment_policy.clone();
    expected.codex_linux_sandbox_exe = turn.codex_linux_sandbox_exe.clone();
    #[allow(deprecated)]
    {
        expected.cwd = turn.cwd.clone();
    }
    expected
        .permissions
        .approval_policy
        .set(AskForApproval::OnRequest)
        .expect("approval policy set");
    expected
        .permissions
        .set_permission_profile(turn.permission_profile())
        .expect("permission profile set");
    assert_eq!(config, expected);
}
