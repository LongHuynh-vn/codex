use super::*;
use codex_apply_patch::MaybeApplyPatchVerified;
use codex_exec_server::LOCAL_FS;
use codex_model_provider::create_model_provider;
use codex_model_provider_info::GEMINI_PROVIDER_ID;
use codex_model_provider_info::ModelProviderInfo;
use codex_model_provider_info::OPENAI_PROVIDER_ID;
use codex_protocol::permissions::FileSystemSandboxPolicy;
use codex_protocol::protocol::FileChange;
use core_test_support::PathBufExt;
use core_test_support::PathExt;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::Mutex;

use crate::session::tests::make_session_and_context;
use crate::session::turn_context::TurnContext;
use crate::tools::context::ToolInvocation;
use crate::tools::hook_names::HookToolName;
use crate::tools::registry::PostToolUsePayload;
use crate::tools::registry::PreToolUsePayload;
use crate::turn_diff_tracker::TurnDiffTracker;

fn sample_patch() -> &'static str {
    r#"*** Begin Patch
*** Add File: hello.txt
+hello
*** End Patch"#
}

async fn invocation_for_payload(payload: ToolPayload) -> ToolInvocation {
    let (session, turn) = make_session_and_context().await;
    ToolInvocation {
        session: session.into(),
        turn: turn.into(),
        cancellation_token: tokio_util::sync::CancellationToken::new(),
        tracker: Arc::new(Mutex::new(TurnDiffTracker::new())),
        call_id: "call-apply-patch".to_string(),
        tool_name: codex_tools::ToolName::plain("apply_patch"),
        source: crate::tools::context::ToolCallSource::Direct,
        payload,
    }
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

fn use_responses_provider(turn: &mut TurnContext) {
    let provider_info = ModelProviderInfo::create_openai_provider(/*base_url*/ None);
    let mut config = (*turn.config).clone();
    config.model_provider_id = OPENAI_PROVIDER_ID.to_string();
    config.model_provider = provider_info.clone();
    turn.provider = create_model_provider(provider_info, turn.auth_manager.clone());
    turn.config = Arc::new(config);
}

async fn intercept_apply_patch_for_test(
    command: &[String],
    session: Arc<Session>,
    turn: Arc<TurnContext>,
) -> Result<Option<FunctionToolOutput>, FunctionCallError> {
    let turn_environment = turn
        .environments
        .primary()
        .cloned()
        .expect("test turn should have a primary environment");
    let cwd = turn_environment.cwd.clone();
    intercept_apply_patch(
        command,
        &cwd,
        LOCAL_FS.as_ref(),
        turn_environment,
        session,
        turn,
        /*tracker*/ None,
        "call-apply-patch",
        "exec_command",
    )
    .await
}

async fn verification_error_for(command: &[String], cwd: &AbsolutePathBuf) -> String {
    let MaybeApplyPatchVerified::CorrectnessError(error) =
        codex_apply_patch::maybe_parse_apply_patch_verified(
            command,
            cwd,
            LOCAL_FS.as_ref(),
            /*sandbox*/ None,
        )
        .await
    else {
        panic!("expected invalid patch to fail verification");
    };
    format!("apply_patch verification failed: {error}")
}

fn invalid_hunk_command() -> Vec<String> {
    vec![
        "apply_patch".to_string(),
        "*** Begin Patch\n*** Frobnicate File: foo\n*** End Patch".to_string(),
    ]
}

fn unsupported_apply_patch_command() -> Vec<String> {
    vec![
        "zsh".to_string(),
        "-lc".to_string(),
        "apply_patch extra-arg <<'PATCH'\n*** Begin Patch\n*** End Patch\nPATCH".to_string(),
    ]
}

#[tokio::test]
async fn pre_tool_use_payload_uses_freeform_patch_input() {
    let patch = sample_patch();
    let payload = ToolPayload::Custom {
        input: patch.to_string(),
    };
    let invocation = invocation_for_payload(payload).await;
    let handler = ApplyPatchHandler::default();

    assert_eq!(
        handler.pre_tool_use_payload(&invocation),
        Some(PreToolUsePayload {
            tool_name: HookToolName::apply_patch(),
            tool_input: json!({ "command": patch }),
        })
    );
}

#[tokio::test]
async fn post_tool_use_payload_uses_patch_input_and_tool_output() {
    let patch = sample_patch();
    let payload = ToolPayload::Custom {
        input: patch.to_string(),
    };
    let invocation = invocation_for_payload(payload).await;
    let output = ApplyPatchToolOutput::from_text("Success. Updated files.".to_string());
    let handler = ApplyPatchHandler::default();

    assert_eq!(
        handler.post_tool_use_payload(&invocation, &output),
        Some(PostToolUsePayload {
            tool_name: HookToolName::apply_patch(),
            tool_use_id: "call-apply-patch".to_string(),
            tool_input: json!({ "command": patch }),
            tool_response: json!("Success. Updated files."),
        })
    );
}

#[tokio::test]
async fn intercepted_apply_patch_adds_retry_guidance_for_gemini_verification_errors() {
    let (session, mut turn) = make_session_and_context().await;
    use_gemini_provider(&mut turn);
    let command = invalid_hunk_command();
    let cwd = turn
        .environments
        .primary()
        .expect("test turn should have a primary environment")
        .cwd
        .clone();
    let expected = format!(
        "{}\n\n{GEMINI_APPLY_PATCH_RETRY_GUIDANCE}",
        verification_error_for(&command, &cwd).await
    );

    let error =
        match intercept_apply_patch_for_test(&command, Arc::new(session), Arc::new(turn)).await {
            Err(error) => error,
            Ok(_) => panic!("invalid Gemini patch should return a model error"),
        };

    assert_eq!(error, FunctionCallError::RespondToModel(expected));
}

#[tokio::test]
async fn intercepted_apply_patch_keeps_responses_verification_errors_unchanged() {
    let (session, mut turn) = make_session_and_context().await;
    use_responses_provider(&mut turn);
    let command = invalid_hunk_command();
    let cwd = turn
        .environments
        .primary()
        .expect("test turn should have a primary environment")
        .cwd
        .clone();
    let expected = verification_error_for(&command, &cwd).await;

    let error =
        match intercept_apply_patch_for_test(&command, Arc::new(session), Arc::new(turn)).await {
            Err(error) => error,
            Ok(_) => panic!("invalid Responses patch should return a model error"),
        };

    assert_eq!(error, FunctionCallError::RespondToModel(expected));
}

#[tokio::test]
async fn intercepted_apply_patch_blocks_unsupported_gemini_shell_form() {
    let (session, mut turn) = make_session_and_context().await;
    use_gemini_provider(&mut turn);
    let command = unsupported_apply_patch_command();

    let error =
        match intercept_apply_patch_for_test(&command, Arc::new(session), Arc::new(turn)).await {
            Err(error) => error,
            Ok(_) => panic!("unsupported Gemini apply_patch form should not fall through"),
        };

    assert_eq!(
        error,
        FunctionCallError::RespondToModel(format!(
            "apply_patch command is not in a supported form. Use the documented apply_patch heredoc form.\n\n{GEMINI_APPLY_PATCH_RETRY_GUIDANCE}"
        ))
    );
}

#[tokio::test]
async fn intercepted_apply_patch_keeps_unsupported_responses_shell_form_falling_through() {
    let (session, mut turn) = make_session_and_context().await;
    use_responses_provider(&mut turn);
    let command = unsupported_apply_patch_command();

    assert!(
        intercept_apply_patch_for_test(&command, Arc::new(session), Arc::new(turn))
            .await
            .expect("Responses apply_patch form should fall through")
            .is_none()
    );
}

#[tokio::test]
async fn intercepted_apply_patch_keeps_non_attempt_gemini_shell_command_falling_through() {
    let (session, mut turn) = make_session_and_context().await;
    use_gemini_provider(&mut turn);
    let command = vec![
        "zsh".to_string(),
        "-lc".to_string(),
        "echo apply_patch".to_string(),
    ];

    assert!(
        intercept_apply_patch_for_test(&command, Arc::new(session), Arc::new(turn))
            .await
            .expect("non-apply_patch command should fall through")
            .is_none()
    );
}

#[test]
fn gemini_unparsed_apply_patch_errors_are_actionable() {
    let command = unsupported_apply_patch_command();
    assert_eq!(
        apply_patch_unparsed_command_error(
            WireApi::GeminiNative,
            &command,
            Some("FailedToParsePatchIntoAst"),
        ),
        Some(format!(
            "apply_patch command could not be parsed: FailedToParsePatchIntoAst. Use the documented apply_patch heredoc form.\n\n{GEMINI_APPLY_PATCH_RETRY_GUIDANCE}"
        ))
    );
    assert_eq!(
        apply_patch_unparsed_command_error(WireApi::Responses, &command, None),
        None
    );
}

#[test]
fn diff_consumer_streams_apply_patch_changes() {
    let mut consumer = ApplyPatchArgumentDiffConsumer::default();
    assert!(
        consumer
            .push_delta("call-1".to_string(), "*** Begin Patch\n")
            .is_none()
    );

    let event = consumer
        .push_delta("call-1".to_string(), "*** Add File: hello.txt\n+hello")
        .expect("progress event");
    assert_eq!(
        (event.call_id, event.changes),
        (
            "call-1".to_string(),
            HashMap::from([(
                PathBuf::from("hello.txt"),
                FileChange::Add {
                    content: String::new(),
                },
            )]),
        )
    );

    assert!(
        consumer
            .push_delta("call-1".to_string(), "\n+world")
            .is_none()
    );
    assert!(
        consumer
            .push_delta("call-1".to_string(), "\n*** End Patch")
            .is_none()
    );

    let event = consumer
        .finish_update_on_complete()
        .expect("finish parser")
        .expect("progress event");
    assert_eq!(
        (event.call_id, event.changes),
        (
            "call-1".to_string(),
            HashMap::from([(
                PathBuf::from("hello.txt"),
                FileChange::Add {
                    content: "hello\nworld\n".to_string(),
                },
            )]),
        )
    );
}

#[test]
fn diff_consumer_streams_apply_patch_changes_with_environment_header() {
    let mut consumer = ApplyPatchArgumentDiffConsumer::default();
    assert!(
        consumer
            .push_delta(
                "call-1".to_string(),
                "*** Begin Patch\n*** Environment ID: remote\n",
            )
            .is_none()
    );

    let event = consumer
        .push_delta("call-1".to_string(), "*** Add File: hello.txt\n+hello")
        .expect("progress event");
    assert_eq!(
        event.changes,
        HashMap::from([(
            PathBuf::from("hello.txt"),
            FileChange::Add {
                content: String::new(),
            },
        )])
    );
}

#[test]
fn diff_consumer_sends_next_update_after_buffer_interval() {
    let mut consumer = ApplyPatchArgumentDiffConsumer::default();
    consumer.push_delta("call-1".to_string(), "*** Begin Patch\n");
    let first = consumer
        .push_delta("call-1".to_string(), "*** Add File: hello.txt\n+hello")
        .expect("first progress event");
    assert_eq!(
        first.changes,
        HashMap::from([(
            PathBuf::from("hello.txt"),
            FileChange::Add {
                content: String::new(),
            },
        )])
    );

    consumer.last_sent_at =
        Some(std::time::Instant::now() - APPLY_PATCH_ARGUMENT_DIFF_BUFFER_INTERVAL);
    let second = consumer
        .push_delta("call-1".to_string(), "\n+world")
        .expect("second progress event");
    assert_eq!(
        second.changes,
        HashMap::from([(
            PathBuf::from("hello.txt"),
            FileChange::Add {
                content: "hello\n".to_string(),
            },
        )])
    );
}

#[test]
fn reconcile_environment_id_requires_selection_when_enabled() {
    assert_eq!(
        require_environment_id(Some("remote"), /*allow_environment_id*/ false),
        Err(FunctionCallError::RespondToModel(
            "apply_patch environment selection is unavailable for this turn".to_string(),
        ))
    );
    assert_eq!(
        require_environment_id(
            /*parsed_environment_id*/ None, /*allow_environment_id*/ true
        ),
        Ok(None)
    );
}

#[tokio::test]
async fn approval_keys_include_move_destination() {
    let tmp = TempDir::new().expect("tmp");
    let cwd_path = tmp.path();
    let cwd = cwd_path.abs();
    std::fs::create_dir_all(cwd_path.join("old")).expect("create old dir");
    std::fs::create_dir_all(cwd_path.join("renamed/dir")).expect("create dest dir");
    std::fs::write(cwd_path.join("old/name.txt"), "old content\n").expect("write old file");
    let patch = r#"*** Begin Patch
*** Update File: old/name.txt
*** Move to: renamed/dir/name.txt
@@
-old content
+new content
*** End Patch"#;
    let argv = vec!["apply_patch".to_string(), patch.to_string()];
    let action = match codex_apply_patch::maybe_parse_apply_patch_verified(
        &argv,
        &cwd,
        LOCAL_FS.as_ref(),
        /*sandbox*/ None,
    )
    .await
    {
        MaybeApplyPatchVerified::Body(action) => action,
        other => panic!("expected patch body, got: {other:?}"),
    };

    let keys = file_paths_for_action(&action);
    assert_eq!(keys.len(), 2);
}

#[test]
fn write_permissions_for_paths_skip_dirs_already_writable_under_workspace_root() {
    let tmp = TempDir::new().expect("tmp");
    let cwd_path = tmp.path();
    let cwd = cwd_path.abs();
    let nested = cwd_path.join("nested");
    std::fs::create_dir_all(&nested).expect("create nested dir");
    let file_path = AbsolutePathBuf::try_from(nested.join("file.txt"))
        .expect("nested file path should be absolute");
    let sandbox_policy = FileSystemSandboxPolicy::workspace_write(
        &[],
        /*exclude_tmpdir_env_var*/ true,
        /*exclude_slash_tmp*/ false,
    );

    let permissions = write_permissions_for_paths(&[file_path], &sandbox_policy, &cwd);

    assert_eq!(permissions, None);
}

#[test]
fn write_permissions_for_paths_keep_dirs_outside_workspace_root() {
    let tmp = TempDir::new().expect("tmp");
    let cwd = tmp.path().join("workspace");
    let outside = tmp.path().join("outside");
    std::fs::create_dir_all(&cwd).expect("create cwd");
    std::fs::create_dir_all(&outside).expect("create outside dir");
    let file_path = AbsolutePathBuf::try_from(outside.join("file.txt"))
        .expect("outside file path should be absolute");
    let cwd_abs = cwd.abs();
    let sandbox_policy = FileSystemSandboxPolicy::workspace_write(
        &[],
        /*exclude_tmpdir_env_var*/ true,
        /*exclude_slash_tmp*/ true,
    );

    let permissions = write_permissions_for_paths(&[file_path], &sandbox_policy, &cwd_abs);
    let expected_outside =
        dunce::simplified(&outside.canonicalize().expect("canonicalize outside dir")).abs();

    assert_eq!(
        permissions
            .and_then(|profile| profile.file_system)
            .and_then(|fs| fs.legacy_read_write_roots())
            .and_then(|(_read, write)| write),
        Some(vec![expected_outside])
    );
}
