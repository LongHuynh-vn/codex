//! O27 Lever 2 / 2b: the Gemini child-only `complete_task` tool.
//!
//! A spawned Gemini sub-agent finalizes its task by calling `complete_task` with a bounded `result`.
//! The handler validates and caps the result, then stashes it on the per-turn `TurnState`; `run_turn`
//! reads that stash immediately after the sampling request and finalizes the child turn with the
//! result as `last_agent_message` (an explicit completion that overrides the implicit turn-end
//! message). Gemini spawned children that do not call `complete_task` get one grace turn; if they
//! still omit it, `run_turn` synthesizes a bounded failed completion from recovered fallback text.

use super::*;
use crate::tools::context::FunctionToolOutput;
use crate::tools::handlers::multi_agents_spec::create_complete_task_tool_v2;
use codex_tools::ToolSpec;
use serde::Deserialize;

/// Hard cap on the bytes of a `complete_task` result we stash as the child's completion. Bounds the
/// report at the source (no model call) so the parent never re-inflates its context with an
/// oversized child report — the structural bound that lets Lever 2 retire the post-hoc summarize in 2b.
const MAX_COMPLETE_TASK_RESULT_BYTES: usize = 12_000;
const COMPLETE_TASK_TRUNCATION_MARKER: &str = "…[truncated at 12000 bytes]";

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CompleteTaskStatus {
    #[default]
    Completed,
    Partial,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompleteTaskResult {
    pub(crate) status: CompleteTaskStatus,
    pub(crate) result: String,
}

impl CompleteTaskResult {
    pub(crate) fn new(status: CompleteTaskStatus, result: &str) -> Self {
        Self {
            status,
            result: cap_result_bytes(result),
        }
    }
}

pub(crate) struct Handler;

#[async_trait::async_trait]
impl ToolExecutor<ToolInvocation> for Handler {
    fn tool_name(&self) -> ToolName {
        ToolName::plain("complete_task")
    }

    fn spec(&self) -> ToolSpec {
        create_complete_task_tool_v2()
    }

    async fn handle(
        &self,
        invocation: ToolInvocation,
    ) -> Result<Box<dyn crate::tools::context::ToolOutput>, FunctionCallError> {
        let arguments = function_arguments(invocation.payload.clone())?;
        let args: CompleteTaskArgs = parse_arguments(&arguments)?;
        handle_complete_task(invocation, args)
            .await
            .map(boxed_tool_output)
    }
}

impl CoreToolRuntime for Handler {
    fn matches_kind(&self, payload: &ToolPayload) -> bool {
        matches!(payload, ToolPayload::Function { .. })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompleteTaskArgs {
    result: String,
    #[serde(default)]
    status: CompleteTaskStatus,
}

async fn handle_complete_task(
    invocation: ToolInvocation,
    args: CompleteTaskArgs,
) -> Result<FunctionToolOutput, FunctionCallError> {
    let ToolInvocation { session, turn, .. } = invocation;

    // Reject an empty report so the model retries instead of "completing" with nothing. A rejected
    // call sets no stash, so the run_turn override does not fire and the turn continues.
    if args.result.trim().is_empty() {
        return Err(FunctionCallError::RespondToModel(
            "complete_task requires a non-empty `result` containing your final report.".to_string(),
        ));
    }

    let result = CompleteTaskResult::new(args.status, &args.result);

    // Stash the bounded result for run_turn to pick up (mirrors the O17 send_message-to-root stash).
    if let Some(turn_state) = session
        .input_queue
        .turn_state_for_sub_id(&session.active_turn, &turn.sub_id)
        .await
    {
        turn_state
            .lock()
            .await
            .gemini_spawned_subagent_complete_task_result = Some(result);
    }

    Ok(FunctionToolOutput::from_text(String::new(), Some(true)))
}

/// Cap `result` at `MAX_COMPLETE_TASK_RESULT_BYTES`, truncating at a UTF-8 char boundary and
/// appending a marker. Leaves room for the marker so the returned string stays within budget.
fn cap_result_bytes(result: &str) -> String {
    if result.len() <= MAX_COMPLETE_TASK_RESULT_BYTES {
        return result.to_string();
    }
    let budget =
        MAX_COMPLETE_TASK_RESULT_BYTES.saturating_sub(COMPLETE_TASK_TRUNCATION_MARKER.len());
    let mut end = budget.min(result.len());
    while end > 0 && !result.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}{}", &result[..end], COMPLETE_TASK_TRUNCATION_MARKER)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn status_defaults_to_completed() {
        let args: CompleteTaskArgs = serde_json::from_value(json!({
            "result": "all done"
        }))
        .expect("missing status should default");

        assert_eq!(args.status, CompleteTaskStatus::Completed);
    }

    #[test]
    fn status_accepts_completed_partial_and_failed() {
        for (raw, expected) in [
            ("completed", CompleteTaskStatus::Completed),
            ("partial", CompleteTaskStatus::Partial),
            ("failed", CompleteTaskStatus::Failed),
        ] {
            let args: CompleteTaskArgs = serde_json::from_value(json!({
                "result": "report",
                "status": raw,
            }))
            .expect("valid status should parse");

            assert_eq!(args.status, expected);
        }
    }

    #[test]
    fn invalid_status_is_rejected() {
        let err = serde_json::from_value::<CompleteTaskArgs>(json!({
            "result": "report",
            "status": "blocked",
        }))
        .expect_err("invalid status should fail deserialization");

        assert!(err.to_string().contains("unknown variant `blocked`"));
    }

    #[test]
    fn short_result_is_not_truncated() {
        let result = "a concise report";
        assert_eq!(cap_result_bytes(result), result);
        assert!(!cap_result_bytes(result).contains(COMPLETE_TASK_TRUNCATION_MARKER));
    }

    #[test]
    fn oversized_result_is_truncated_with_marker_and_stays_within_budget() {
        let result = "x".repeat(20_000);
        let capped = cap_result_bytes(&result);
        assert!(capped.ends_with(COMPLETE_TASK_TRUNCATION_MARKER));
        assert!(
            capped.len() <= MAX_COMPLETE_TASK_RESULT_BYTES,
            "capped len {} exceeded budget {MAX_COMPLETE_TASK_RESULT_BYTES}",
            capped.len()
        );
    }

    #[test]
    fn truncation_respects_utf8_char_boundaries() {
        // Multi-byte chars (é = 2 bytes) force the boundary walk-back; the result must stay valid UTF-8.
        let result = "é".repeat(20_000);
        let capped = cap_result_bytes(&result);
        assert!(capped.ends_with(COMPLETE_TASK_TRUNCATION_MARKER));
        assert!(capped.len() <= MAX_COMPLETE_TASK_RESULT_BYTES);
        // `capped` being a String already guarantees valid UTF-8; assert the body decodes cleanly too.
        let body = capped
            .strip_suffix(COMPLETE_TASK_TRUNCATION_MARKER)
            .unwrap();
        assert!(body.chars().all(|c| c == 'é'));
    }
}
