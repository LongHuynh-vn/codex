#![allow(clippy::expect_used)]

use super::gemini_native_phase2::GeminiRequestLog;
use super::gemini_native_phase2::gemini_builder;
use super::gemini_native_phase2::gemini_function_call_sse;
use super::gemini_native_phase2::gemini_text_sse;
use super::gemini_native_phase2::mount_gemini_sse_once_match;
use super::subagent_notifications::body_contains;
use super::subagent_notifications::body_marker_count;
use super::subagent_notifications::wait_for_spawned_thread_id;
use anyhow::Result;
use codex_core::CodexThread;
use codex_features::Feature;
use codex_protocol::ThreadId;
use codex_protocol::protocol::AgentStatus;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodexHarness;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;
use tokio::time::sleep;
use wiremock::Match;
use wiremock::Mock;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path_regex;

const REMINDER_MARKER: &str = "Your delegated sub-agent token budget is nearly exhausted";
const EXHAUSTED_MARKER: &str = "Your delegated sub-agent token budget is exhausted";
const COMPLETE_TASK_REPORT: &str = "bounded child findings";

#[derive(Clone, Copy)]
struct GeminiUsage {
    prompt: i64,
    candidates: i64,
    thoughts: i64,
}

struct BodyContains(&'static str);

impl Match for BodyContains {
    fn matches(&self, request: &wiremock::Request) -> bool {
        body_contains(request, self.0)
    }
}

fn gemini_function_call_with_usage(
    name: &str,
    args: Value,
    thought_signature: &str,
    usage: GeminiUsage,
) -> String {
    let total = usage
        .prompt
        .saturating_add(usage.candidates)
        .saturating_add(usage.thoughts);
    let chunk = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "name": name,
                        "args": args,
                    },
                    "thoughtSignature": thought_signature,
                }],
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": usage.prompt,
            "candidatesTokenCount": usage.candidates,
            "thoughtsTokenCount": usage.thoughts,
            "totalTokenCount": total,
        },
    });
    format!("data: {chunk}\n\ndata: [DONE]\n\n")
}

async fn setup_spawned_child(
    parent_prompt: &'static str,
    child_prompt: &'static str,
    task_name: &'static str,
    child_token_budget: i64,
) -> Result<(TestCodexHarness, GeminiRequestLog)> {
    let harness = TestCodexHarness::with_builder(gemini_builder().with_config(move |config| {
        config
            .features
            .enable(Feature::MultiAgentV2)
            .expect("test config should allow feature update");
        config.multi_agent_v2.child_token_budget = child_token_budget;
        config.model_provider.request_max_retries = Some(0);
        config.model_provider.stream_max_retries = Some(0);
    }))
    .await?;
    let server = harness.server();

    let _parent_initial = mount_gemini_sse_once_match(
        server,
        move |req| {
            body_contains(req, parent_prompt) && !body_contains(req, r#""functionResponse""#)
        },
        gemini_function_call_sse(
            "spawn_agent",
            json!({
                "message": child_prompt,
                "task_name": task_name,
            }),
            Some("sig-budget-parent-spawn"),
        ),
        /*response_delay*/ None,
    )
    .await;
    let _parent_after_spawn = mount_gemini_sse_once_match(
        server,
        move |req| {
            body_contains(req, parent_prompt)
                && body_contains(req, r#""functionResponse""#)
                && body_contains(req, r#""name":"spawn_agent""#)
                && !body_contains(req, "<subagent_notification>")
        },
        gemini_text_sse("parent waiting for the budgeted child"),
        /*response_delay*/ None,
    )
    .await;
    let synthesis = mount_gemini_sse_once_match(
        server,
        |req| body_contains(req, "<subagent_notification>"),
        gemini_text_sse("parent synthesized the budgeted child report"),
        /*response_delay*/ None,
    )
    .await;

    Ok((harness, synthesis))
}

async fn spawned_child_thread(harness: &TestCodexHarness) -> Result<Arc<CodexThread>> {
    let spawned_id = wait_for_spawned_thread_id(harness.test()).await?;
    let spawned_id = ThreadId::from_string(&spawned_id).map_err(anyhow::Error::msg)?;
    Ok(harness.test().thread_manager.get_thread(spawned_id).await?)
}

async fn wait_for_child_completion(harness: &TestCodexHarness) -> Result<AgentStatus> {
    let child_thread = spawned_child_thread(harness).await?;
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let status = child_thread.agent_status().await;
        if matches!(status, AgentStatus::Completed(_)) {
            return Ok(status);
        }
        if Instant::now() >= deadline {
            anyhow::bail!("timed out waiting for child completion, last status: {status:?}");
        }
        sleep(Duration::from_millis(10)).await;
    }
}

async fn wait_for_synthesis(log: &GeminiRequestLog) -> Result<Value> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(request) = log.requests().into_iter().next() {
            return Ok(request);
        }
        if Instant::now() >= deadline {
            anyhow::bail!("timed out waiting for parent synthesis request");
        }
        sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reminder_fires_once_at_twenty_percent_remaining_then_complete_task_wins() -> Result<()> {
    skip_if_no_network!(Ok(()));

    const PARENT_PROMPT: &str = "spawn a child that finishes after its budget reminder";
    const CHILD_PROMPT: &str = "child: perform two bounded steps, then finish on reminder";
    let (harness, synthesis) = setup_spawned_child(
        PARENT_PROMPT,
        CHILD_PROMPT,
        "reminder_worker",
        /*child_token_budget*/ 30_000,
    )
    .await?;
    let server = harness.server();
    let usage = GeminiUsage {
        prompt: 10_000,
        candidates: 1_000,
        thoughts: 1_000,
    };

    let first_step = mount_gemini_sse_once_match(
        server,
        move |req| body_contains(req, CHILD_PROMPT) && !body_contains(req, r#""functionResponse""#),
        gemini_function_call_with_usage("list_agents", json!({}), "sig-reminder-step-1", usage),
        /*response_delay*/ None,
    )
    .await;
    let second_step = mount_gemini_sse_once_match(
        server,
        move |req| {
            body_contains(req, CHILD_PROMPT)
                && body_contains(req, r#""name":"list_agents""#)
                && body_contains(req, r#""functionResponse""#)
                && !body_contains(req, REMINDER_MARKER)
        },
        gemini_function_call_with_usage("list_agents", json!({}), "sig-reminder-step-2", usage),
        /*response_delay*/ None,
    )
    .await;
    let reminder = mount_gemini_sse_once_match(
        server,
        |req| body_marker_count(req, REMINDER_MARKER) == 1,
        gemini_function_call_sse(
            "complete_task",
            json!({ "result": COMPLETE_TASK_REPORT, "status": "completed" }),
            Some("sig-reminder-complete-task"),
        ),
        /*response_delay*/ None,
    )
    .await;

    harness.test().submit_turn(PARENT_PROMPT).await?;
    assert_eq!(
        wait_for_child_completion(&harness).await?,
        AgentStatus::Completed(Some(format!(
            "Sub-agent task completed:\n{COMPLETE_TASK_REPORT}"
        )))
    );
    assert_eq!(first_step.requests().len(), 1);
    assert_eq!(second_step.requests().len(), 1);
    let reminder_requests = reminder.requests();
    assert_eq!(reminder_requests.len(), 1);
    let reminder_body = reminder_requests[0].to_string();
    assert_eq!(reminder_body.matches(REMINDER_MARKER).count(), 1);
    assert!(!reminder_body.contains(EXHAUSTED_MARKER));

    let synthesis_request = wait_for_synthesis(&synthesis).await?.to_string();
    assert!(synthesis_request.contains(COMPLETE_TASK_REPORT));
    assert!(!synthesis_request.contains(r#""completed":null"#));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exhaustion_gets_one_grounding_free_sample_then_forces_non_null_failure() -> Result<()> {
    skip_if_no_network!(Ok(()));

    const PARENT_PROMPT: &str = "spawn a child that exhausts its token budget";
    const CHILD_PROMPT: &str = "child: keep working until the token budget stops you";
    let (harness, synthesis) = setup_spawned_child(
        PARENT_PROMPT,
        CHILD_PROMPT,
        "exhausted_worker",
        /*child_token_budget*/ 20_000,
    )
    .await?;
    let server = harness.server();
    let usage = GeminiUsage {
        prompt: 10_000,
        candidates: 1_000,
        thoughts: 1_000,
    };

    let first_step = mount_gemini_sse_once_match(
        server,
        move |req| body_contains(req, CHILD_PROMPT) && !body_contains(req, r#""functionResponse""#),
        gemini_function_call_with_usage("list_agents", json!({}), "sig-exhaust-step-1", usage),
        /*response_delay*/ None,
    )
    .await;
    let crossing_step = mount_gemini_sse_once_match(
        server,
        move |req| {
            body_contains(req, CHILD_PROMPT)
                && body_contains(req, r#""name":"list_agents""#)
                && body_contains(req, r#""functionResponse""#)
                && !body_contains(req, EXHAUSTED_MARKER)
        },
        gemini_function_call_with_usage("list_agents", json!({}), "sig-exhaust-step-2", usage),
        /*response_delay*/ None,
    )
    .await;
    let grace = mount_gemini_sse_once_match(
        server,
        |req| body_marker_count(req, EXHAUSTED_MARKER) == 1,
        gemini_function_call_with_usage("list_agents", json!({}), "sig-exhaust-grace", usage),
        /*response_delay*/ None,
    )
    .await;

    harness.test().submit_turn(PARENT_PROMPT).await?;
    let status = wait_for_child_completion(&harness).await?;
    let AgentStatus::Completed(Some(report)) = status else {
        anyhow::bail!("exhausted child must complete with a non-null report: {status:?}");
    };
    assert!(report.starts_with("Sub-agent task failed:\nChild token budget exhausted after "));
    assert!(report.contains("limit 20000"));
    assert_eq!(first_step.requests().len(), 1);
    assert_eq!(crossing_step.requests().len(), 1);
    let grace_requests = grace.requests();
    assert_eq!(grace_requests.len(), 1);
    let grace_body = grace_requests[0].to_string();
    assert_eq!(grace_body.matches(EXHAUSTED_MARKER).count(), 1);
    assert!(!grace_body.contains("googleSearch"));

    let synthesis_request = wait_for_synthesis(&synthesis).await?.to_string();
    assert!(synthesis_request.contains("Child token budget exhausted"));
    assert!(!synthesis_request.contains(r#""completed":null"#));

    let child_thread = spawned_child_thread(&harness).await?;
    const REENGAGEMENT_PROMPT: &str = "re-engage the already exhausted child";
    let reengaged_turn_id = child_thread
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: REENGAGEMENT_PROMPT.to_string(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    let reengaged_completion = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let event = child_thread.next_event().await?;
            if let EventMsg::TurnComplete(completion) = event.msg
                && completion.turn_id == reengaged_turn_id
            {
                return anyhow::Ok(completion);
            }
        }
    })
    .await??;
    let Some(reengaged_report) = reengaged_completion.last_agent_message else {
        anyhow::bail!("re-engaged exhausted child must return a non-null failed result");
    };
    assert!(
        reengaged_report.starts_with("Sub-agent task failed:\nChild token budget exhausted after ")
    );
    sleep(Duration::from_millis(100)).await;
    let requests_after_reengagement = server.received_requests().await.unwrap_or_default();
    assert_eq!(
        requests_after_reengagement
            .iter()
            .filter(|request| body_contains(request, REENGAGEMENT_PROMPT))
            .count(),
        0,
        "re-engaging a force-completed child must issue zero child sampling requests"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn errored_exhaustion_grace_forces_non_null_failure_without_retry() -> Result<()> {
    skip_if_no_network!(Ok(()));

    const PARENT_PROMPT: &str = "spawn a child whose budget grace request errors";
    const CHILD_PROMPT: &str = "child: work until the budget grace request fails";
    let (harness, synthesis) = setup_spawned_child(
        PARENT_PROMPT,
        CHILD_PROMPT,
        "errored_grace_worker",
        /*child_token_budget*/ 20_000,
    )
    .await?;
    let server = harness.server();
    let usage = GeminiUsage {
        prompt: 10_000,
        candidates: 1_000,
        thoughts: 1_000,
    };

    let first_step = mount_gemini_sse_once_match(
        server,
        move |req| body_contains(req, CHILD_PROMPT) && !body_contains(req, r#""functionResponse""#),
        gemini_function_call_with_usage("list_agents", json!({}), "sig-error-step-1", usage),
        /*response_delay*/ None,
    )
    .await;
    let crossing_step = mount_gemini_sse_once_match(
        server,
        move |req| {
            body_contains(req, CHILD_PROMPT)
                && body_contains(req, r#""functionResponse""#)
                && !body_contains(req, EXHAUSTED_MARKER)
        },
        gemini_function_call_with_usage("list_agents", json!({}), "sig-error-step-2", usage),
        /*response_delay*/ None,
    )
    .await;
    let _failed_grace = Mock::given(method("POST"))
        .and(path_regex(
            ".*/models/gemini-3\\.5-flash:streamGenerateContent$",
        ))
        .and(BodyContains(EXHAUSTED_MARKER))
        .respond_with(
            ResponseTemplate::new(500)
                .insert_header("content-type", "application/json")
                .set_body_json(json!({
                    "error": {"message": "synthetic budget grace failure"}
                })),
        )
        .expect(1)
        .mount_as_scoped(server)
        .await;

    harness.test().submit_turn(PARENT_PROMPT).await?;
    let status = wait_for_child_completion(&harness).await?;
    let AgentStatus::Completed(Some(report)) = status else {
        anyhow::bail!("errored grace must still produce a non-null report: {status:?}");
    };
    assert!(report.starts_with("Sub-agent task failed:\nChild token budget exhausted after "));
    assert_eq!(first_step.requests().len(), 1);
    assert_eq!(crossing_step.requests().len(), 1);
    let grace_request_count = server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .filter(|request| body_contains(request, EXHAUSTED_MARKER))
        .count();
    assert_eq!(grace_request_count, 1);

    let synthesis_request = wait_for_synthesis(&synthesis).await?.to_string();
    assert!(synthesis_request.contains("Child token budget exhausted"));
    assert!(!synthesis_request.contains(r#""completed":null"#));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zero_budget_disables_reminder_and_exhaustion_for_large_child_usage() -> Result<()> {
    skip_if_no_network!(Ok(()));

    const PARENT_PROMPT: &str = "spawn a child with disabled token budgeting";
    const CHILD_PROMPT: &str = "child: finish normally despite high reported usage";
    let (harness, synthesis) = setup_spawned_child(
        PARENT_PROMPT,
        CHILD_PROMPT,
        "disabled_budget_worker",
        /*child_token_budget*/ 0,
    )
    .await?;
    let server = harness.server();

    let first_step = mount_gemini_sse_once_match(
        server,
        move |req| body_contains(req, CHILD_PROMPT) && !body_contains(req, r#""functionResponse""#),
        gemini_function_call_with_usage(
            "list_agents",
            json!({}),
            "sig-disabled-step",
            GeminiUsage {
                prompt: 25_000,
                candidates: 2_500,
                thoughts: 2_500,
            },
        ),
        /*response_delay*/ None,
    )
    .await;
    let completion = mount_gemini_sse_once_match(
        server,
        move |req| {
            body_contains(req, CHILD_PROMPT)
                && body_contains(req, r#""functionResponse""#)
                && !body_contains(req, REMINDER_MARKER)
                && !body_contains(req, EXHAUSTED_MARKER)
        },
        gemini_function_call_sse(
            "complete_task",
            json!({ "result": COMPLETE_TASK_REPORT, "status": "completed" }),
            Some("sig-disabled-complete-task"),
        ),
        /*response_delay*/ None,
    )
    .await;

    harness.test().submit_turn(PARENT_PROMPT).await?;
    assert_eq!(
        wait_for_child_completion(&harness).await?,
        AgentStatus::Completed(Some(format!(
            "Sub-agent task completed:\n{COMPLETE_TASK_REPORT}"
        )))
    );
    assert_eq!(first_step.requests().len(), 1);
    let completion_requests = completion.requests();
    assert_eq!(completion_requests.len(), 1);
    let completion_body = completion_requests[0].to_string();
    assert!(!completion_body.contains(REMINDER_MARKER));
    assert!(!completion_body.contains(EXHAUSTED_MARKER));
    let synthesis_request = wait_for_synthesis(&synthesis).await?.to_string();
    assert!(!synthesis_request.contains(r#""completed":null"#));

    Ok(())
}
