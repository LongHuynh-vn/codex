//! Goal-free structural recovery for Gemini-native root orchestration (O31).
//!
//! Covers two failure modes observed with the Gemini-native auto-goal
//! subsystem (`crate::goals`) disabled: a parent that receives all delegated
//! child reports but never emits a final answer (idle forever), and a parent
//! that keeps re-verifying delegated work with no token ceiling.
//!
//! This module is dormant whenever `GoalRuntimeState::auto_armed_orchestration_goal_id`
//! is armed for a session (see `orchestration_dormant` below) — the goal
//! subsystem owns idle/runaway recovery in that case. It exists so the goal
//! subsystem can eventually be deleted without leaving those two failure
//! modes uncovered.
//!
//! State here is in-memory only and does not survive process resume, exactly
//! like the goal runtime's O27 markers (`crate::goals`, `auto_armed_orchestration_goal_id`
//! / `orchestration_answer_emitted_goal_id`) — no reconstruction is built.
//!
//! A handful of small pure predicates/constants are deliberately duplicated
//! here rather than imported from their originals, because a separate,
//! already-shelved change deletes those originals when the goal subsystem is
//! removed. Duplicating them means this module keeps compiling after that
//! removal lands, instead of breaking alongside the goal machinery it was
//! built to eventually replace. Each duplicate below says which original it
//! mirrors.

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

/// Mirrors `tools::handlers::multi_agents_v2::spawn::GEMINI_ORCHESTRATION_AUTO_GOAL_TOKEN_BUDGET`
/// (250_000). Duplicated, not imported: the goal-removal change deletes that
/// constant.
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

/// Duplicate of `agent::status::is_terminal_with_delivery` — kept independent
/// so this module keeps compiling once the goal-removal change deletes that
/// function. Keep in sync if child-terminal semantics change.
fn is_child_terminal_with_delivery(status: &AgentStatus) -> bool {
    matches!(
        status,
        AgentStatus::Completed(Some(_)) | AgentStatus::Errored(_) | AgentStatus::Shutdown
    )
}

/// Duplicate of `tools::handlers::multi_agents_v2::spawn::is_root_orchestrator_source`
/// — kept independent so this module keeps compiling once the goal-removal
/// change deletes that function.
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

/// Duplicate of `goals::should_ignore_goal_for_mode` — kept independent
/// because the original is private to `goals.rs` (not `pub(crate)`), and this
/// module intentionally makes zero edits to `goals.rs`, including
/// visibility-only ones.
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
    /// Master dormancy switch: while the Gemini-native auto-goal subsystem has
    /// successfully armed a goal for this session, it owns idle/runaway
    /// recovery and this module must stay fully inert.
    async fn orchestration_dormant(&self) -> bool {
        self.goal_runtime
            .auto_armed_orchestration_goal_id
            .lock()
            .await
            .is_some()
    }

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

    /// Re-arm chokepoint: called alongside `mark_reengaged_child_this_turn` at
    /// every point that re-engages a child (spawn_agent, send_message,
    /// followup_task). Clears the answer marker and the nudge latch so a new
    /// delegation epoch gets a fresh chance to nudge. Does not touch
    /// `steer_multiples_fired` — see its doc comment.
    pub(crate) async fn mark_orchestration_reengaged(&self, turn_context: &TurnContext) {
        if !is_gemini_native_root(turn_context) {
            return;
        }
        let mut inner = self.orchestration_runtime.inner.lock().await;
        inner.answer_emitted_since_reengagement = false;
        inner.nudge_fired = false;
    }

    /// Change 1 (covers S1): if all delegated children are terminal-with-
    /// delivery and no answer has been given since the last re-engagement,
    /// inject one internal-context nudge to make the parent emit its
    /// consolidated final answer now. Fires at most once per delegation
    /// epoch.
    pub(crate) async fn maybe_nudge_gemini_orchestration_idle(
        self: &Arc<Self>,
        turn_context: &TurnContext,
        emitted_final_answer: bool,
        reengaged_child_this_turn: bool,
    ) {
        if !is_gemini_native_root(turn_context) {
            return;
        }
        if is_plan_mode(turn_context.collaboration_mode.mode) {
            tracing::debug!("orchestration nudge blocked: plan_mode");
            return;
        }
        if self.orchestration_dormant().await {
            tracing::debug!("orchestration nudge blocked: dormant_goal");
            return;
        }

        {
            let mut inner = self.orchestration_runtime.inner.lock().await;
            if !inner.started {
                tracing::debug!(
                    reengaged_child_this_turn,
                    "orchestration nudge blocked: not_started"
                );
                return;
            }
            if emitted_final_answer {
                // Deliberately asymmetric with the goal system's
                // G0-before-G3 ordering: mid-turn re-engagement already
                // clears this marker at the spawn/send_message chokepoint, so
                // an answer observed at turn end counts as "since the last
                // re-engagement." If a rare answer-then-spawn turn makes that
                // marker stale, suppressing this nudge errs toward today's
                // stall behavior; the opposite error from removing the
                // re-engagement gate is a spurious nudge, bounded to one wasted
                // turn by the latch. We prefer the bounded error.
                inner.answer_emitted_since_reengagement = true;
                tracing::debug!(
                    reengaged_child_this_turn,
                    "orchestration nudge blocked: answer_emitted"
                );
                return;
            }
            if inner.answer_emitted_since_reengagement {
                tracing::debug!(
                    reengaged_child_this_turn,
                    "orchestration nudge blocked: answer_emitted"
                );
                return;
            }
            if inner.nudge_fired {
                tracing::debug!(
                    reengaged_child_this_turn,
                    "orchestration nudge blocked: nudge_already_fired"
                );
                return;
            }
        }

        if self.input_queue.has_trigger_turn_mailbox_items().await {
            tracing::debug!(
                reengaged_child_this_turn,
                "orchestration nudge blocked: mailbox_pending"
            );
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
        // closed), matching `goals::maybe_auto_complete_gemini_orchestration_goal`'s
        // G7 semantics exactly. The never-spawned-root case is already
        // excluded above by `started`.
        for (child_thread_id, _metadata) in &children {
            let status: AgentStatus = self
                .services
                .agent_control
                .get_status(*child_thread_id)
                .await;
            if !is_child_terminal_with_delivery(&status) {
                tracing::debug!(
                    reengaged_child_this_turn,
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
                tracing::debug!(
                    reengaged_child_this_turn,
                    "orchestration nudge blocked: nudge_already_fired"
                );
                return;
            }
            inner.nudge_fired = true;
        }

        tracing::debug!(reengaged_child_this_turn, "orchestration nudge fired");
        let item = orchestration_context_item(NUDGE_PROMPT.to_string());
        if self.try_start_turn_if_idle(vec![item]).await.is_err() {
            tracing::debug!("orchestration nudge forfeited for this generation: mailbox/turn race");
        }
    }

    /// Change 2 (covers S2): once cumulative token usage since the first
    /// spawn crosses another full `ORCHESTRATION_TOKEN_CEILING` increment,
    /// inject a wrap-up steer into the active turn. Re-fires at each
    /// additional increment, not just once, so a long-running runaway keeps
    /// getting pressure to wrap up rather than going silent after one nudge.
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
        if self.orchestration_dormant().await {
            tracing::debug!("orchestration ceiling blocked: dormant_goal");
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
