//! Goal-free structural recovery for Gemini-native root orchestration (O31).
//!
//! Covers two failure modes observed in Gemini-native orchestration: a parent
//! that receives all delegated child reports but never emits a final answer
//! (idle forever), and a parent that keeps re-verifying delegated work with no
//! token ceiling.
//!
//! State here is in-memory only and does not survive process resume; no
//! reconstruction is built.
//!
//! A handful of small pure predicates/constants are deliberately duplicated
//! here rather than imported from the removed auto-goal machinery. That keeps
//! this module independent from the old crutch it replaces.

use crate::context::ContextualUserFragment;
use crate::context::InternalContextSource;
use crate::context::InternalModelContextFragment;
use crate::goals::goal_token_delta_for_usage;
use crate::session::session::Session;
use crate::session::turn_context::TurnContext;
use codex_model_provider_info::WireApi;
use codex_protocol::config_types::ModeKind;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::AgentStatus;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::TokenUsage;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Token ceiling for Gemini-native orchestration wrap-up steering.
const ORCHESTRATION_TOKEN_CEILING: i64 = 250_000;

const NUDGE_PROMPT: &str = "All delegated children have reported. Emit the consolidated \
final answer to the user NOW, in the final channel. Do not re-verify or re-do delegated \
work; treat child reports as authoritative.";

const CEILING_PROMPT: &str = "Token budget for this orchestration is exhausted. Stop \
verifying and re-checking. Synthesize what you already have and emit the consolidated \
final answer in the final channel now.";

#[derive(Default)]
pub(crate) struct OrchestrationRuntimeInner {
    /// Set on the first successful Gemini-native root `spawn_agent`. Never
    /// cleared for the life of the session.
    pub(crate) started: bool,
    /// Total token usage snapshot captured alongside `started`. Never reset,
    /// including across re-engagements — the ceiling tracks cumulative spend
    /// for the whole orchestration, not per delegation epoch.
    pub(crate) token_baseline: Option<TokenUsage>,
    /// Whether a final-channel answer has been emitted since the last child
    /// re-engagement (spawn/send_message/followup_task). Cleared on
    /// re-engagement; set when an idle turn-end observes a final answer.
    pub(crate) answer_emitted_since_reengagement: bool,
    /// One-shot latch: the idle nudge may fire at most once per delegation
    /// epoch. Cleared on re-engagement.
    pub(crate) nudge_fired: bool,
    /// Watermark of how many full `ORCHESTRATION_TOKEN_CEILING` increments
    /// have already been steered against, computed from the token delta since
    /// `token_baseline`. Never cleared by re-engagement: it tracks cumulative
    /// spend from the fixed first-spawn baseline, so the ceiling keeps
    /// re-firing every additional 250K crossed for as long as the
    /// orchestration runs, rather than firing once and going silent.
    pub(crate) steer_multiples_fired: i64,
}

/// Session-owned, goal-free orchestration recovery state. See module docs.
pub(crate) struct OrchestrationRuntimeState {
    pub(crate) inner: Mutex<OrchestrationRuntimeInner>,
}

impl OrchestrationRuntimeState {
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(OrchestrationRuntimeInner::default()),
        }
    }
}

/// Child states that count as terminal with a delivered result for
/// orchestration recovery.
fn is_child_terminal_with_delivery(status: &AgentStatus) -> bool {
    matches!(
        status,
        AgentStatus::Completed(Some(_)) | AgentStatus::Errored(_) | AgentStatus::Shutdown
    )
}

/// Session sources that represent a root orchestrator.
fn is_root_orchestrator(session_source: &SessionSource) -> bool {
    match session_source {
        SessionSource::Cli
        | SessionSource::VSCode
        | SessionSource::Exec
        | SessionSource::Mcp
        | SessionSource::Custom(_)
        | SessionSource::Unknown => true,
        SessionSource::SubAgent(_) => session_source
            .get_agent_path()
            .is_some_and(|path| path.is_root()),
        SessionSource::Internal(_) => false,
    }
}

/// Local mode predicate for orchestration recovery. Kept local because the goal
/// runtime's equivalent predicate is private to `goals.rs`.
fn is_plan_mode(mode: ModeKind) -> bool {
    mode == ModeKind::Plan
}

fn is_gemini_native_root(turn_context: &TurnContext) -> bool {
    turn_context.provider.info().wire_api == WireApi::GeminiNative
        && is_root_orchestrator(&turn_context.session_source)
}

fn orchestration_context_item(prompt: String) -> ResponseItem {
    ContextualUserFragment::into(InternalModelContextFragment::new(
        InternalContextSource::from_static("orchestration"),
        prompt,
    ))
}

/// Field-wise `(current - baseline).max(0)`, mirroring the turn-usage delta
/// pattern already established at `tasks::mod::on_task_finished`.
fn token_usage_delta(current: &TokenUsage, baseline: &TokenUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: (current.input_tokens - baseline.input_tokens).max(0),
        cached_input_tokens: (current.cached_input_tokens - baseline.cached_input_tokens).max(0),
        output_tokens: (current.output_tokens - baseline.output_tokens).max(0),
        reasoning_output_tokens: (current.reasoning_output_tokens
            - baseline.reasoning_output_tokens)
            .max(0),
        total_tokens: (current.total_tokens - baseline.total_tokens).max(0),
    }
}

impl Session {
    /// Captures the first-spawn token baseline and flips `started` on the
    /// FIRST successful Gemini-native root `spawn_agent` only. Idempotent:
    /// later spawns are no-ops here.
    pub(crate) async fn mark_orchestration_started_if_first(&self, turn_context: &TurnContext) {
        if !is_gemini_native_root(turn_context) {
            return;
        }
        let baseline = self.total_token_usage().await.unwrap_or_default();
        let mut inner = self.orchestration_runtime.inner.lock().await;
        if inner.started {
            return;
        }
        inner.started = true;
        inner.token_baseline = Some(baseline);
    }

    /// Re-arm chokepoint: called at every point that re-engages a child
    /// (spawn_agent, send_message, followup_task). Clears the answer marker and
    /// the nudge latch so a new delegation epoch gets a fresh chance to nudge.
    /// Does not touch
    /// `steer_multiples_fired` — see its doc comment.
    pub(crate) async fn mark_orchestration_reengaged(&self, turn_context: &TurnContext) {
        if !is_gemini_native_root(turn_context) {
            return;
        }
        let mut inner = self.orchestration_runtime.inner.lock().await;
        inner.answer_emitted_since_reengagement = false;
        inner.nudge_fired = false;
    }

    /// If all delegated children are terminal-with-delivery and no answer has
    /// been given since the last re-engagement, inject one internal-context
    /// nudge to make the parent emit its consolidated final answer now. Fires
    /// at most once per delegation epoch.
    pub(crate) async fn maybe_nudge_gemini_orchestration_idle(
        self: &Arc<Self>,
        turn_context: &TurnContext,
        emitted_final_answer: bool,
    ) {
        if !is_gemini_native_root(turn_context) {
            return;
        }
        if is_plan_mode(turn_context.collaboration_mode.mode) {
            tracing::debug!("orchestration nudge blocked: plan_mode");
            return;
        }

        {
            let mut inner = self.orchestration_runtime.inner.lock().await;
            if !inner.started {
                tracing::debug!("orchestration nudge blocked: not_started");
                return;
            }
            if emitted_final_answer {
                // Mid-turn re-engagement clears this marker at the
                // spawn/send_message chokepoint, so an answer observed at turn
                // end counts as "since the last re-engagement." If a rare
                // answer-then-spawn turn makes that marker stale, suppression
                // errs toward the old stall behavior rather than an extra
                // bounded nudge.
                inner.answer_emitted_since_reengagement = true;
                tracing::debug!("orchestration nudge blocked: answer_emitted");
                return;
            }
            if inner.answer_emitted_since_reengagement {
                tracing::debug!("orchestration nudge blocked: answer_emitted");
                return;
            }
            if inner.nudge_fired {
                tracing::debug!("orchestration nudge blocked: nudge_already_fired");
                return;
            }
        }

        if self.input_queue.has_trigger_turn_mailbox_items().await {
            tracing::debug!("orchestration nudge blocked: mailbox_pending");
            return;
        }

        let children = match self
            .services
            .agent_control
            .open_thread_spawn_children(self.thread_id)
            .await
        {
            Ok(children) => children,
            Err(err) => {
                tracing::debug!(
                    "orchestration nudge: failed to enumerate children, skipping: {err}"
                );
                return;
            }
        };
        // An empty list is vacuously delivered (e.g. all children explicitly
        // closed). The never-spawned-root case is already excluded above by
        // `started`.
        for (child_thread_id, _metadata) in &children {
            let status: AgentStatus = self
                .services
                .agent_control
                .get_status(*child_thread_id)
                .await;
            if !is_child_terminal_with_delivery(&status) {
                tracing::debug!(
                    %child_thread_id,
                    ?status,
                    "orchestration nudge blocked: child_not_terminal"
                );
                return;
            }
        }

        {
            let mut inner = self.orchestration_runtime.inner.lock().await;
            if inner.nudge_fired {
                // Another call raced ahead during the awaits above.
                tracing::debug!("orchestration nudge blocked: nudge_already_fired");
                return;
            }
            inner.nudge_fired = true;
        }

        tracing::debug!("orchestration nudge fired");
        let item = orchestration_context_item(NUDGE_PROMPT.to_string());
        if self.try_start_turn_if_idle(vec![item]).await.is_err() {
            tracing::debug!("orchestration nudge forfeited for this generation: mailbox/turn race");
        }
    }

    /// Once cumulative token usage since the first spawn crosses another full
    /// `ORCHESTRATION_TOKEN_CEILING` increment, inject a wrap-up steer into the
    /// active turn. Re-fires at each additional increment, not just once, so a
    /// long-running runaway keeps getting pressure to wrap up rather than going
    /// silent after one nudge.
    #[expect(
        clippy::await_holding_invalid_type,
        reason = "the ceiling check-then-steer sequence must remain atomic across the \
                  injection await so two concurrent tool-finishes can't both claim the \
                  same ceiling multiple, mirroring goals::account_thread_goal_progress's \
                  own accounting_permit hold across its equivalent injection"
    )]
    pub(crate) async fn maybe_steer_gemini_orchestration_ceiling(
        &self,
        turn_context: &TurnContext,
    ) {
        if !is_gemini_native_root(turn_context) {
            return;
        }
        if is_plan_mode(turn_context.collaboration_mode.mode) {
            tracing::debug!("orchestration ceiling blocked: plan_mode");
            return;
        }

        let mut inner = self.orchestration_runtime.inner.lock().await;
        if !inner.started {
            tracing::debug!("orchestration ceiling blocked: not_started");
            return;
        }
        let baseline = inner.token_baseline.clone().unwrap_or_default();
        let current = self.total_token_usage().await.unwrap_or_default();
        let delta = goal_token_delta_for_usage(&token_usage_delta(&current, &baseline));
        let target_multiple = delta / ORCHESTRATION_TOKEN_CEILING;
        if target_multiple <= inner.steer_multiples_fired {
            return;
        }

        let item = orchestration_context_item(CEILING_PROMPT.to_string());
        if self.inject_if_running(vec![item]).await.is_ok() {
            inner.steer_multiples_fired = target_multiple;
        }
    }
}
