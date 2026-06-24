//! Core support for persisted thread goals.
//!
//! This module bridges core sessions and the state-db goal table. It validates
//! goal mutations, converts between state and protocol shapes, emits goal-update
//! events, and owns helper hooks used by goal lifecycle behavior.

use crate::StateDbHandle;
use crate::agent::status::is_terminal_with_delivery;
use crate::context::ContextualUserFragment;
use crate::context::InternalContextSource;
use crate::context::InternalModelContextFragment;
use crate::session::TurnInput;
use crate::session::session::Session;
use crate::session::turn_context::TurnContext;
use crate::state::ActiveTurn;
use crate::state::TurnState;
use crate::tasks::RegularTask;
use crate::tools::handlers::goal_spec::UPDATE_GOAL_TOOL_NAME;
use crate::tools::handlers::multi_agents_v2::is_root_orchestrator_source;
use anyhow::Context;
use codex_features::Feature;
use codex_model_provider_info::WireApi;
use codex_otel::GOAL_BLOCKED_METRIC;
use codex_otel::GOAL_BUDGET_LIMITED_METRIC;
use codex_otel::GOAL_COMPLETED_METRIC;
use codex_otel::GOAL_CREATED_METRIC;
use codex_otel::GOAL_DURATION_SECONDS_METRIC;
use codex_otel::GOAL_RESUMED_METRIC;
use codex_otel::GOAL_TOKEN_COUNT_METRIC;
use codex_otel::GOAL_USAGE_LIMITED_METRIC;
use codex_prompts::budget_limit_prompt;
use codex_prompts::continuation_prompt;
use codex_prompts::objective_updated_prompt;
use codex_prompts::orchestration_continuation_prompt;
use codex_protocol::ThreadId;
use codex_protocol::config_types::ModeKind;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::AgentStatus;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::ThreadGoal;
use codex_protocol::protocol::ThreadGoalStatus;
use codex_protocol::protocol::ThreadGoalUpdatedEvent;
use codex_protocol::protocol::TokenUsage;
use codex_protocol::protocol::validate_thread_goal_objective;
use codex_rollout::state_db::reconcile_rollout;
use codex_thread_store::LocalThreadStore;
use futures::future::BoxFuture;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::Mutex;
use tokio::sync::Semaphore;
use tokio::sync::SemaphorePermit;

pub(crate) struct SetGoalRequest {
    pub(crate) objective: Option<String>,
    pub(crate) status: Option<ThreadGoalStatus>,
    pub(crate) token_budget: Option<Option<i64>>,
}

pub(crate) struct CreateGoalRequest {
    pub(crate) objective: String,
    pub(crate) token_budget: Option<i64>,
}

#[derive(Clone, Copy)]
enum BudgetLimitSteering {
    Allowed,
    Suppressed,
}

#[derive(Clone, Copy)]
enum TerminalMetricEmission {
    Emit,
    Suppress,
}

/// Describes whether an external goal mutation created a new logical goal or
/// updated an existing one.
#[derive(Clone)]
pub enum ExternalGoalPreviousStatus {
    NewGoal,
    Existing(ExternalGoalPreviousGoal),
}

#[derive(Clone)]
pub struct ExternalGoalPreviousGoal {
    goal_id: String,
    status: codex_state::ThreadGoalStatus,
    objective: String,
}

impl From<&codex_state::ThreadGoal> for ExternalGoalPreviousStatus {
    fn from(goal: &codex_state::ThreadGoal) -> Self {
        Self::Existing(ExternalGoalPreviousGoal::from(goal))
    }
}

impl From<&codex_state::ThreadGoal> for ExternalGoalPreviousGoal {
    fn from(goal: &codex_state::ThreadGoal) -> Self {
        Self {
            goal_id: goal.goal_id.clone(),
            status: goal.status,
            objective: goal.objective.clone(),
        }
    }
}

/// Runtime effects for an externally persisted goal mutation.
#[derive(Clone)]
pub struct ExternalGoalSet {
    pub goal: codex_state::ThreadGoal,
    pub previous_status: ExternalGoalPreviousStatus,
}

/// Runtime lifecycle events that can affect goal accounting, scheduling, or
/// model-visible steering.
///
/// Callers report the session event they observed; this module owns the policy
/// for how that event changes goal runtime state.
pub(crate) enum GoalRuntimeEvent<'a> {
    TurnStarted {
        turn_context: &'a TurnContext,
        token_usage: TokenUsage,
    },
    ToolCompleted {
        turn_context: &'a TurnContext,
        tool_name: &'a str,
    },
    ToolCompletedGoal {
        turn_context: &'a TurnContext,
    },
    TurnFinished {
        turn_context: &'a TurnContext,
        turn_completed: bool,
    },
    MaybeContinueIfIdle,
    TaskAborted {
        turn_context: Option<&'a TurnContext>,
    },
    UsageLimitReached {
        turn_context: &'a TurnContext,
    },
    ExternalMutationStarting,
    ExternalSet {
        external_set: ExternalGoalSet,
    },
    ExternalClear,
    ThreadResumed,
}

pub(crate) struct GoalRuntimeState {
    pub(crate) state_db: Mutex<Option<StateDbHandle>>,
    pub(crate) budget_limit_reported_goal_id: Mutex<Option<String>>,
    /// Goal id of the goal that was auto-armed by a Gemini-native root
    /// orchestrator on its first `spawn_agent`. Provenance marker: only a goal
    /// recorded here may be structurally auto-completed or receive the
    /// orchestration continuation prompt — a user-created goal is never
    /// auto-completed. In-memory runtime only (no schema change); set once at
    /// auto-arm and keyed by exact goal id, so a replaced/different goal never
    /// matches.
    pub(crate) auto_armed_orchestration_goal_id: Mutex<Option<String>>,
    /// O27: goal id of the auto-armed orchestration goal for which the parent has
    /// emitted a consolidated final-channel answer at least once *since the last
    /// child re-engagement*. Durable across the idle goal-continuation turns so
    /// the structural auto-complete can fire on a later clean idle turn (the
    /// synthesize-then-coordinate sequence), not only the turn that emitted the
    /// answer. Set when a clean answer turn is observed; cleared whenever a child
    /// is re-engaged (spawn/followup/send) so a re-opened child re-requires a
    /// fresh synthesis. In-memory only; keyed by exact goal id.
    pub(crate) orchestration_answer_emitted_goal_id: Mutex<Option<String>>,
    accounting_lock: Semaphore,
    accounting: Mutex<GoalAccountingSnapshot>,
    pub(crate) continuation_lock: Semaphore,
}

struct GoalContinuationCandidate {
    goal_id: String,
    items: Vec<ResponseItem>,
}

impl GoalRuntimeState {
    pub(crate) fn new() -> Self {
        Self {
            state_db: Mutex::new(None),
            budget_limit_reported_goal_id: Mutex::new(None),
            auto_armed_orchestration_goal_id: Mutex::new(None),
            orchestration_answer_emitted_goal_id: Mutex::new(None),
            accounting_lock: Semaphore::new(/*permits*/ 1),
            accounting: Mutex::new(GoalAccountingSnapshot::new()),
            continuation_lock: Semaphore::new(/*permits*/ 1),
        }
    }
}

#[derive(Debug)]
struct GoalAccountingSnapshot {
    turn: Option<GoalTurnAccountingSnapshot>,
    wall_clock: GoalWallClockAccountingSnapshot,
}

#[derive(Debug)]
struct GoalTurnAccountingSnapshot {
    turn_id: String,
    last_accounted_token_usage: TokenUsage,
    active_goal_id: Option<String>,
}

impl GoalRuntimeState {
    async fn accounting_permit(&self) -> anyhow::Result<SemaphorePermit<'_>> {
        self.accounting_lock
            .acquire()
            .await
            .context("goal accounting semaphore closed")
    }
}

impl GoalAccountingSnapshot {
    fn new() -> Self {
        Self {
            turn: None,
            wall_clock: GoalWallClockAccountingSnapshot::new(),
        }
    }
}

impl GoalTurnAccountingSnapshot {
    fn new(turn_id: impl Into<String>, token_usage: TokenUsage) -> Self {
        Self {
            turn_id: turn_id.into(),
            last_accounted_token_usage: token_usage,
            active_goal_id: None,
        }
    }

    fn mark_active_goal(&mut self, goal_id: impl Into<String>) {
        self.active_goal_id = Some(goal_id.into());
    }

    fn active_this_turn(&self) -> bool {
        self.active_goal_id.is_some()
    }

    fn active_goal_id(&self) -> Option<String> {
        self.active_goal_id.clone()
    }

    fn clear_active_goal(&mut self) {
        self.active_goal_id = None;
    }

    fn reset_baseline(&mut self, token_usage: TokenUsage) {
        self.last_accounted_token_usage = token_usage;
    }

    fn token_delta_since_last_accounting(&self, current: &TokenUsage) -> i64 {
        let last = &self.last_accounted_token_usage;
        let delta = TokenUsage {
            input_tokens: current.input_tokens.saturating_sub(last.input_tokens),
            cached_input_tokens: current
                .cached_input_tokens
                .saturating_sub(last.cached_input_tokens),
            output_tokens: current.output_tokens.saturating_sub(last.output_tokens),
            reasoning_output_tokens: current
                .reasoning_output_tokens
                .saturating_sub(last.reasoning_output_tokens),
            total_tokens: current.total_tokens.saturating_sub(last.total_tokens),
        };
        goal_token_delta_for_usage(&delta)
    }

    fn mark_accounted(&mut self, current: TokenUsage) {
        self.last_accounted_token_usage = current;
    }
}

#[derive(Debug)]
struct GoalWallClockAccountingSnapshot {
    last_accounted_at: Instant,
    active_goal_id: Option<String>,
}

impl GoalWallClockAccountingSnapshot {
    fn new() -> Self {
        Self {
            last_accounted_at: Instant::now(),
            active_goal_id: None,
        }
    }

    fn time_delta_since_last_accounting(&self) -> i64 {
        let last = self.last_accounted_at;
        i64::try_from(last.elapsed().as_secs()).unwrap_or(i64::MAX)
    }

    fn mark_accounted(&mut self, accounted_seconds: i64) {
        if accounted_seconds <= 0 {
            return;
        }
        let advance = Duration::from_secs(u64::try_from(accounted_seconds).unwrap_or(u64::MAX));
        self.last_accounted_at = self
            .last_accounted_at
            .checked_add(advance)
            .unwrap_or_else(Instant::now);
    }

    fn reset_baseline(&mut self) {
        self.last_accounted_at = Instant::now();
    }

    fn mark_active_goal(&mut self, goal_id: impl Into<String>) {
        let goal_id = goal_id.into();
        if self.active_goal_id.as_deref() != Some(goal_id.as_str()) {
            self.reset_baseline();
            self.active_goal_id = Some(goal_id);
        }
    }

    fn clear_active_goal(&mut self) {
        self.active_goal_id = None;
        self.reset_baseline();
    }

    fn active_goal_id(&self) -> Option<String> {
        self.active_goal_id.clone()
    }
}

impl Session {
    /// Applies runtime policy for a goal lifecycle event.
    ///
    /// Goal data methods validate and persist state; this dispatcher owns the
    /// cross-cutting runtime behavior: plan mode ignores continuations, turn
    /// starts capture the active goal and token baseline, tool completions
    /// account usage and may inject budget steering, completion accounting
    /// suppresses that steering, external mutations account best-effort before
    /// changing state, thread resumes restore runtime state for already-active
    /// goals, explicit maybe-continue events
    /// start idle goal continuation turns, and continuation turns with no counted
    /// autonomous activity suppress the next automatic continuation until
    /// user/tool/external activity resets it.
    pub(crate) fn goal_runtime_apply<'a>(
        self: &'a Arc<Self>,
        event: GoalRuntimeEvent<'a>,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
        match event {
            GoalRuntimeEvent::TurnStarted {
                turn_context,
                token_usage,
            } => Box::pin(async move {
                self.mark_thread_goal_turn_started(turn_context, token_usage)
                    .await;
                Ok(())
            }),
            GoalRuntimeEvent::ToolCompleted {
                turn_context,
                tool_name,
            } => Box::pin(async move {
                if tool_name != UPDATE_GOAL_TOOL_NAME {
                    self.account_thread_goal_progress(
                        turn_context,
                        BudgetLimitSteering::Allowed,
                        TerminalMetricEmission::Emit,
                    )
                    .await?;
                }
                Ok(())
            }),
            GoalRuntimeEvent::ToolCompletedGoal { turn_context } => Box::pin(async move {
                self.account_thread_goal_progress(
                    turn_context,
                    BudgetLimitSteering::Suppressed,
                    TerminalMetricEmission::Suppress,
                )
                .await?;
                Ok(())
            }),
            GoalRuntimeEvent::TurnFinished {
                turn_context,
                turn_completed,
            } => Box::pin(async move {
                self.finish_thread_goal_turn(turn_context, turn_completed)
                    .await;
                Ok(())
            }),
            GoalRuntimeEvent::MaybeContinueIfIdle => Box::pin(async move {
                self.maybe_continue_goal_if_idle_runtime().await;
                Ok(())
            }),
            GoalRuntimeEvent::TaskAborted { turn_context } => Box::pin(async move {
                self.handle_thread_goal_task_abort(turn_context).await;
                Ok(())
            }),
            GoalRuntimeEvent::UsageLimitReached { turn_context } => Box::pin(async move {
                self.usage_limit_active_thread_goal_for_turn(turn_context)
                    .await?;
                Ok(())
            }),
            GoalRuntimeEvent::ExternalMutationStarting => Box::pin(async move {
                if let Err(err) = self.account_thread_goal_before_external_mutation().await {
                    tracing::warn!(
                        "failed to account thread goal progress before external mutation: {err}"
                    );
                }
                Ok(())
            }),
            GoalRuntimeEvent::ExternalSet { external_set } => Box::pin(async move {
                self.apply_external_thread_goal_status(external_set).await;
                Ok(())
            }),
            GoalRuntimeEvent::ExternalClear => Box::pin(async move {
                self.clear_stopped_thread_goal_runtime_state().await;
                Ok(())
            }),
            GoalRuntimeEvent::ThreadResumed => Box::pin(async move {
                self.restore_thread_goal_runtime_after_resume().await?;
                Ok(())
            }),
        }
    }

    pub(crate) async fn get_thread_goal(&self) -> anyhow::Result<Option<ThreadGoal>> {
        if !self.enabled(Feature::Goals) {
            anyhow::bail!("goals feature is disabled");
        }

        let state_db = self.require_state_db_for_thread_goals().await?;
        state_db
            .thread_goals()
            .get_thread_goal(self.thread_id)
            .await
            .map(|goal| goal.map(protocol_goal_from_state))
    }

    /// Records the goal auto-armed by a Gemini-native root orchestrator as the
    /// orchestration goal eligible for structural auto-completion and the
    /// orchestration continuation prompt. Reads the freshly-created goal's id
    /// from state (the protocol `ThreadGoal` returned by `create_thread_goal`
    /// does not carry it). Set once at auto-arm; keyed by the exact goal id so a
    /// replaced/user goal never matches.
    pub(crate) async fn mark_auto_armed_orchestration_goal(&self) {
        let goal_id = match self.state_db_for_thread_goals().await {
            Ok(Some(state_db)) => match state_db
                .thread_goals()
                .get_thread_goal(self.thread_id)
                .await
            {
                Ok(Some(goal)) => goal.goal_id,
                Ok(None) => return,
                Err(err) => {
                    tracing::debug!("failed to read auto-armed orchestration goal id: {err}");
                    return;
                }
            },
            Ok(None) => return,
            Err(err) => {
                tracing::debug!(
                    "failed to open state db to record auto-armed orchestration goal: {err}"
                );
                return;
            }
        };
        *self
            .goal_runtime
            .auto_armed_orchestration_goal_id
            .lock()
            .await = Some(goal_id);
    }

    /// Flags the current turn as having invoked a child re-engagement tool
    /// (`spawn_agent`, `followup_task`, `send_message`). Read at turn-end to
    /// prevent auto-completing the orchestration goal on a turn that may have
    /// re-opened a child. No-op when there is no active turn for `turn_context`.
    ///
    /// O27: re-engaging a child also invalidates the durable "answer emitted"
    /// marker. A re-opened child must be chased and a *fresh* consolidated answer
    /// produced before the goal can auto-complete; clearing here (the single
    /// chokepoint both `spawn_agent` and the message tools route through, set
    /// before any early return) closes the followup-reopen race durably.
    pub(crate) async fn mark_reengaged_child_this_turn(&self, turn_context: &TurnContext) {
        *self
            .goal_runtime
            .orchestration_answer_emitted_goal_id
            .lock()
            .await = None;
        if let Some(turn_state) = self
            .input_queue
            .turn_state_for_sub_id(&self.active_turn, &turn_context.sub_id)
            .await
        {
            turn_state.lock().await.reengaged_child_this_turn = true;
        }
    }

    pub(crate) async fn set_thread_goal(
        &self,
        turn_context: &TurnContext,
        request: SetGoalRequest,
    ) -> anyhow::Result<ThreadGoal> {
        if !self.enabled(Feature::Goals) {
            anyhow::bail!("goals feature is disabled");
        }

        let SetGoalRequest {
            objective,
            status,
            token_budget,
        } = request;
        validate_goal_budget(token_budget.flatten())?;
        let state_db = self.require_state_db_for_thread_goals().await?;
        let objective = objective.map(|objective| objective.trim().to_string());
        if let Some(objective) = objective.as_deref()
            && let Err(err) = validate_thread_goal_objective(objective)
        {
            anyhow::bail!("{err}");
        }

        self.account_thread_goal_wall_clock_usage(
            &state_db,
            codex_state::GoalAccountingMode::ActiveOnly,
            TerminalMetricEmission::Emit,
        )
        .await?;
        let mut replacing_goal = false;
        let previous_status;
        let goal = if let Some(objective) = objective.as_deref() {
            let existing_goal = state_db
                .thread_goals()
                .get_thread_goal(self.thread_id)
                .await?;
            previous_status = existing_goal.as_ref().map(|goal| goal.status);
            if let Some(existing_goal) = existing_goal.as_ref() {
                state_db
                    .thread_goals()
                    .update_thread_goal(
                        self.thread_id,
                        codex_state::GoalUpdate {
                            objective: Some(objective.to_string()),
                            status: status.map(state_goal_status_from_protocol),
                            token_budget,
                            expected_goal_id: Some(existing_goal.goal_id.clone()),
                        },
                    )
                    .await?
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "cannot update goal for thread {}: no goal exists",
                            self.thread_id
                        )
                    })?
            } else {
                replacing_goal = true;
                state_db
                    .thread_goals()
                    .replace_thread_goal(
                        self.thread_id,
                        objective,
                        status
                            .map(state_goal_status_from_protocol)
                            .unwrap_or(codex_state::ThreadGoalStatus::Active),
                        token_budget.flatten(),
                    )
                    .await?
            }
        } else {
            let existing_goal = state_db
                .thread_goals()
                .get_thread_goal(self.thread_id)
                .await?;
            previous_status = existing_goal.as_ref().map(|goal| goal.status);
            let expected_goal_id = existing_goal.map(|goal| goal.goal_id);
            let status = status.map(state_goal_status_from_protocol);
            state_db
                .thread_goals()
                .update_thread_goal(
                    self.thread_id,
                    codex_state::GoalUpdate {
                        objective: None,
                        status,
                        token_budget,
                        expected_goal_id,
                    },
                )
                .await?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "cannot update goal for thread {}: no goal exists",
                        self.thread_id
                    )
                })?
        };

        if objective.is_some() {
            set_thread_preview_from_goal_objective(
                &state_db,
                self.thread_id,
                goal.objective.as_str(),
            )
            .await;
        }
        let goal_status = goal.status;
        let goal_id = goal.goal_id.clone();
        let previous_status_for_goal = if replacing_goal {
            None
        } else {
            previous_status
        };
        if replacing_goal {
            self.emit_goal_created_metric();
        }
        self.emit_goal_resumed_metric_if_status_changed(previous_status_for_goal, goal_status);
        self.emit_goal_terminal_metrics_if_status_changed(previous_status_for_goal, &goal);
        let goal = protocol_goal_from_state(goal);
        *self.goal_runtime.budget_limit_reported_goal_id.lock().await = None;
        let newly_active_goal = goal_status == codex_state::ThreadGoalStatus::Active
            && (replacing_goal
                || previous_status
                    .is_some_and(|status| status != codex_state::ThreadGoalStatus::Active));
        if newly_active_goal {
            let current_token_usage = self.total_token_usage().await.unwrap_or_default();
            self.mark_active_goal_accounting(
                goal_id,
                Some(turn_context.sub_id.clone()),
                current_token_usage,
            )
            .await;
        } else if goal_status != codex_state::ThreadGoalStatus::Active {
            self.clear_active_goal_accounting(turn_context).await;
        }
        self.send_event(
            turn_context,
            EventMsg::ThreadGoalUpdated(ThreadGoalUpdatedEvent {
                thread_id: self.thread_id,
                turn_id: Some(turn_context.sub_id.clone()),
                goal: goal.clone(),
            }),
        )
        .await;
        Ok(goal)
    }

    pub(crate) async fn create_thread_goal(
        &self,
        turn_context: &TurnContext,
        request: CreateGoalRequest,
    ) -> anyhow::Result<ThreadGoal> {
        if !self.enabled(Feature::Goals) {
            anyhow::bail!("goals feature is disabled");
        }

        let CreateGoalRequest {
            objective,
            token_budget,
        } = request;
        validate_goal_budget(token_budget)?;
        let objective = objective.trim();
        validate_thread_goal_objective(objective).map_err(anyhow::Error::msg)?;

        let state_db = self.require_state_db_for_thread_goals().await?;
        self.account_thread_goal_wall_clock_usage(
            &state_db,
            codex_state::GoalAccountingMode::ActiveOnly,
            TerminalMetricEmission::Emit,
        )
        .await?;
        let goal = state_db
            .thread_goals()
            .insert_thread_goal(
                self.thread_id,
                objective,
                codex_state::ThreadGoalStatus::Active,
                token_budget,
            )
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "cannot create a new goal because thread {} already has a goal",
                    self.thread_id
                )
            })?;

        set_thread_preview_from_goal_objective(&state_db, self.thread_id, goal.objective.as_str())
            .await;
        let goal_id = goal.goal_id.clone();
        self.emit_goal_created_metric();
        let goal = protocol_goal_from_state(goal);
        *self.goal_runtime.budget_limit_reported_goal_id.lock().await = None;

        let current_token_usage = self.total_token_usage().await.unwrap_or_default();
        self.mark_active_goal_accounting(
            goal_id,
            Some(turn_context.sub_id.clone()),
            current_token_usage,
        )
        .await;

        self.send_event(
            turn_context,
            EventMsg::ThreadGoalUpdated(ThreadGoalUpdatedEvent {
                thread_id: self.thread_id,
                turn_id: Some(turn_context.sub_id.clone()),
                goal: goal.clone(),
            }),
        )
        .await;
        Ok(goal)
    }

    async fn apply_external_thread_goal_status(self: &Arc<Self>, external_set: ExternalGoalSet) {
        let ExternalGoalSet {
            goal,
            previous_status,
        } = external_set;
        let previous_goal = match previous_status {
            ExternalGoalPreviousStatus::NewGoal => None,
            ExternalGoalPreviousStatus::Existing(goal) => Some(goal),
        };
        let replaced_existing_goal = previous_goal
            .as_ref()
            .is_some_and(|previous_goal| previous_goal.goal_id != goal.goal_id);
        if previous_goal.is_none() || replaced_existing_goal {
            self.emit_goal_created_metric();
        }
        let objective_changed = previous_goal
            .as_ref()
            .is_some_and(|previous_goal| previous_goal.objective != goal.objective);
        let previous_status = previous_goal
            .as_ref()
            .and_then(|previous_goal| (!replaced_existing_goal).then_some(previous_goal.status));
        self.emit_goal_resumed_metric_if_status_changed(previous_status, goal.status);
        self.emit_goal_terminal_metrics_if_status_changed(previous_status, &goal);
        let goal_for_steering = objective_changed.then(|| protocol_goal_from_state(goal.clone()));
        let goal_id = goal.goal_id;
        let status = goal.status;
        match status {
            codex_state::ThreadGoalStatus::Active => {
                let turn_id = self
                    .active_turn_context()
                    .await
                    .map(|turn_context| turn_context.sub_id.clone());
                let current_token_usage = self.total_token_usage().await.unwrap_or_default();
                self.mark_active_goal_accounting(goal_id, turn_id, current_token_usage)
                    .await;
                if let Some(goal) = goal_for_steering {
                    let item = goal_context_input_item(objective_updated_prompt(&goal));
                    if self.inject_if_running(vec![item]).await.is_err() {
                        tracing::debug!(
                            "skipping objective-updated goal steering because no turn is active"
                        );
                    }
                }
                self.maybe_continue_goal_if_idle_runtime().await;
            }
            codex_state::ThreadGoalStatus::BudgetLimited => {
                if self.active_turn_context().await.is_none() {
                    self.clear_stopped_thread_goal_runtime_state().await;
                }
            }
            codex_state::ThreadGoalStatus::Paused
            | codex_state::ThreadGoalStatus::Blocked
            | codex_state::ThreadGoalStatus::UsageLimited
            | codex_state::ThreadGoalStatus::Complete => {
                self.clear_stopped_thread_goal_runtime_state().await;
            }
        }
    }

    async fn clear_stopped_thread_goal_runtime_state(&self) {
        *self.goal_runtime.budget_limit_reported_goal_id.lock().await = None;
        let mut accounting = self.goal_runtime.accounting.lock().await;
        if let Some(turn) = accounting.turn.as_mut() {
            turn.clear_active_goal();
        }
        accounting.wall_clock.clear_active_goal();
    }

    async fn clear_active_goal_accounting(&self, turn_context: &TurnContext) {
        let mut accounting = self.goal_runtime.accounting.lock().await;
        if let Some(turn) = accounting.turn.as_mut()
            && turn.turn_id == turn_context.sub_id
        {
            turn.clear_active_goal();
        }
        accounting.wall_clock.clear_active_goal();
    }

    async fn mark_active_goal_accounting(
        &self,
        goal_id: String,
        turn_id: Option<String>,
        token_usage: TokenUsage,
    ) {
        let mut accounting = self.goal_runtime.accounting.lock().await;
        if let Some(turn_id) = turn_id {
            match accounting.turn.as_mut() {
                Some(turn) if turn.turn_id == turn_id => {
                    turn.reset_baseline(token_usage);
                    turn.mark_active_goal(goal_id.clone());
                }
                _ => {
                    let mut turn = GoalTurnAccountingSnapshot::new(turn_id, token_usage);
                    turn.mark_active_goal(goal_id.clone());
                    accounting.turn = Some(turn);
                }
            }
        }
        accounting.wall_clock.mark_active_goal(goal_id);
    }

    fn emit_goal_created_metric(&self) {
        self.services
            .session_telemetry
            .counter(GOAL_CREATED_METRIC, /*inc*/ 1, &[]);
    }

    fn emit_goal_resumed_metric(&self) {
        self.services
            .session_telemetry
            .counter(GOAL_RESUMED_METRIC, /*inc*/ 1, &[]);
    }

    fn emit_goal_resumed_metric_if_status_changed(
        &self,
        previous_status: Option<codex_state::ThreadGoalStatus>,
        goal_status: codex_state::ThreadGoalStatus,
    ) {
        if goal_status == codex_state::ThreadGoalStatus::Active
            && matches!(
                previous_status,
                Some(
                    codex_state::ThreadGoalStatus::Paused
                        | codex_state::ThreadGoalStatus::Blocked
                        | codex_state::ThreadGoalStatus::UsageLimited
                )
            )
        {
            self.emit_goal_resumed_metric();
        }
    }

    fn emit_goal_terminal_metrics_if_status_changed(
        &self,
        previous_status: Option<codex_state::ThreadGoalStatus>,
        goal: &codex_state::ThreadGoal,
    ) {
        if previous_status == Some(goal.status) {
            return;
        }

        let counter = match goal.status {
            codex_state::ThreadGoalStatus::Blocked => GOAL_BLOCKED_METRIC,
            codex_state::ThreadGoalStatus::UsageLimited => GOAL_USAGE_LIMITED_METRIC,
            codex_state::ThreadGoalStatus::BudgetLimited => GOAL_BUDGET_LIMITED_METRIC,
            codex_state::ThreadGoalStatus::Complete => GOAL_COMPLETED_METRIC,
            codex_state::ThreadGoalStatus::Active | codex_state::ThreadGoalStatus::Paused => {
                return;
            }
        };
        let status_tag = [("status", goal.status.as_str())];
        self.services
            .session_telemetry
            .counter(counter, /*inc*/ 1, &[]);
        self.services.session_telemetry.histogram(
            GOAL_TOKEN_COUNT_METRIC,
            goal.tokens_used,
            &status_tag,
        );
        self.services.session_telemetry.histogram(
            GOAL_DURATION_SECONDS_METRIC,
            goal.time_used_seconds,
            &status_tag,
        );
    }

    async fn current_goal_status_for_metrics(
        &self,
        state_db: &StateDbHandle,
        expected_goal_id: Option<&str>,
    ) -> anyhow::Result<Option<codex_state::ThreadGoalStatus>> {
        let goal = state_db
            .thread_goals()
            .get_thread_goal(self.thread_id)
            .await?;
        Ok(goal.and_then(|goal| {
            expected_goal_id
                .is_none_or(|expected_goal_id| goal.goal_id == expected_goal_id)
                .then_some(goal.status)
        }))
    }

    async fn active_turn_context(&self) -> Option<Arc<TurnContext>> {
        let active = self.active_turn.lock().await;
        active
            .as_ref()
            .and_then(|active_turn| active_turn.task.as_ref())
            .map(|task| Arc::clone(&task.turn_context))
    }

    async fn mark_thread_goal_turn_started(
        &self,
        turn_context: &TurnContext,
        token_usage: TokenUsage,
    ) {
        self.goal_runtime.accounting.lock().await.turn = Some(GoalTurnAccountingSnapshot::new(
            turn_context.sub_id.clone(),
            token_usage,
        ));

        if !self.enabled(Feature::Goals) {
            return;
        }
        if should_ignore_goal_for_mode(turn_context.collaboration_mode.mode) {
            self.clear_active_goal_accounting(turn_context).await;
            return;
        }
        let state_db = match self.state_db_for_thread_goals().await {
            Ok(Some(state_db)) => state_db,
            Ok(None) => return,
            Err(err) => {
                tracing::warn!("failed to open state db at turn start: {err}");
                return;
            }
        };
        match state_db
            .thread_goals()
            .get_thread_goal(self.thread_id)
            .await
        {
            Ok(Some(goal))
                if matches!(
                    goal.status,
                    codex_state::ThreadGoalStatus::Active
                        | codex_state::ThreadGoalStatus::BudgetLimited
                ) =>
            {
                let mut accounting = self.goal_runtime.accounting.lock().await;
                if let Some(turn) = accounting.turn.as_mut()
                    && turn.turn_id == turn_context.sub_id
                {
                    turn.mark_active_goal(goal.goal_id.clone());
                }
                accounting.wall_clock.mark_active_goal(goal.goal_id);
            }
            Ok(Some(_)) | Ok(None) => {
                self.goal_runtime
                    .accounting
                    .lock()
                    .await
                    .wall_clock
                    .clear_active_goal();
            }
            Err(err) => {
                tracing::warn!("failed to read thread goal at turn start: {err}");
            }
        }
    }

    async fn clear_reserved_goal_continuation_turn(&self, turn_state: &Arc<Mutex<TurnState>>) {
        let mut active_turn_guard = self.active_turn.lock().await;
        if let Some(active_turn) = active_turn_guard.as_ref()
            && active_turn.task.is_none()
            && Arc::ptr_eq(&active_turn.turn_state, turn_state)
        {
            *active_turn_guard = None;
        }
    }

    async fn finish_thread_goal_turn(
        self: &Arc<Self>,
        turn_context: &TurnContext,
        turn_completed: bool,
    ) {
        if turn_completed
            && let Err(err) = self
                .account_thread_goal_progress(
                    turn_context,
                    BudgetLimitSteering::Suppressed,
                    TerminalMetricEmission::Emit,
                )
                .await
        {
            tracing::warn!("failed to account thread goal progress at turn end: {err}");
        }

        if turn_completed {
            let mut accounting = self.goal_runtime.accounting.lock().await;
            if accounting
                .turn
                .as_ref()
                .is_some_and(|turn| turn.turn_id == turn_context.sub_id)
            {
                accounting.turn = None;
            }
        }
    }

    async fn handle_thread_goal_task_abort(&self, turn_context: Option<&TurnContext>) {
        if let Some(turn_context) = turn_context {
            if let Err(err) = self
                .account_thread_goal_progress(
                    turn_context,
                    BudgetLimitSteering::Suppressed,
                    TerminalMetricEmission::Emit,
                )
                .await
            {
                tracing::warn!("failed to account thread goal progress after abort: {err}");
            }
            let mut accounting = self.goal_runtime.accounting.lock().await;
            if accounting
                .turn
                .as_ref()
                .is_some_and(|turn| turn.turn_id == turn_context.sub_id)
            {
                accounting.turn = None;
            }
        }
    }

    async fn account_thread_goal_progress(
        &self,
        turn_context: &TurnContext,
        budget_limit_steering: BudgetLimitSteering,
        terminal_metric_emission: TerminalMetricEmission,
    ) -> anyhow::Result<()> {
        if !self.enabled(Feature::Goals) {
            return Ok(());
        }
        if should_ignore_goal_for_mode(turn_context.collaboration_mode.mode) {
            return Ok(());
        }
        let Some(state_db) = self.state_db_for_thread_goals().await? else {
            return Ok(());
        };
        let _accounting_permit = self.goal_runtime.accounting_permit().await?;
        let current_token_usage = self.total_token_usage().await.unwrap_or_default();
        let (token_delta, expected_goal_id, time_delta_seconds) = {
            let accounting = self.goal_runtime.accounting.lock().await;
            let Some(turn) = accounting
                .turn
                .as_ref()
                .filter(|turn| turn.turn_id == turn_context.sub_id)
            else {
                return Ok(());
            };
            if !turn.active_this_turn() {
                return Ok(());
            }
            (
                turn.token_delta_since_last_accounting(&current_token_usage),
                turn.active_goal_id(),
                accounting.wall_clock.time_delta_since_last_accounting(),
            )
        };
        if time_delta_seconds == 0 && token_delta <= 0 {
            return Ok(());
        }
        let previous_status = self
            .current_goal_status_for_metrics(&state_db, expected_goal_id.as_deref())
            .await?;
        let outcome = state_db
            .thread_goals()
            .account_thread_goal_usage(
                self.thread_id,
                time_delta_seconds,
                token_delta,
                codex_state::GoalAccountingMode::ActiveOnly,
                expected_goal_id.as_deref(),
            )
            .await?;
        let budget_limit_was_already_reported = {
            let reported_goal_id = self.goal_runtime.budget_limit_reported_goal_id.lock().await;
            expected_goal_id
                .as_deref()
                .is_some_and(|goal_id| reported_goal_id.as_deref() == Some(goal_id))
        };
        let goal = match outcome {
            codex_state::GoalAccountingOutcome::Updated(goal) => {
                let clear_active_goal = match goal.status {
                    codex_state::ThreadGoalStatus::Active => false,
                    codex_state::ThreadGoalStatus::BudgetLimited => {
                        matches!(budget_limit_steering, BudgetLimitSteering::Suppressed)
                    }
                    codex_state::ThreadGoalStatus::Paused
                    | codex_state::ThreadGoalStatus::Blocked
                    | codex_state::ThreadGoalStatus::UsageLimited
                    | codex_state::ThreadGoalStatus::Complete => true,
                };
                {
                    let mut accounting = self.goal_runtime.accounting.lock().await;
                    if let Some(turn) = accounting
                        .turn
                        .as_mut()
                        .filter(|turn| turn.turn_id == turn_context.sub_id)
                    {
                        turn.mark_accounted(current_token_usage);
                        if clear_active_goal {
                            turn.clear_active_goal();
                        }
                    }
                    accounting.wall_clock.mark_accounted(time_delta_seconds);
                    if clear_active_goal {
                        accounting.wall_clock.clear_active_goal();
                    }
                }
                if matches!(terminal_metric_emission, TerminalMetricEmission::Emit) {
                    self.emit_goal_terminal_metrics_if_status_changed(previous_status, &goal);
                }
                goal
            }
            codex_state::GoalAccountingOutcome::Unchanged(_) => return Ok(()),
        };
        let should_steer_budget_limit =
            matches!(budget_limit_steering, BudgetLimitSteering::Allowed)
                && goal.status == codex_state::ThreadGoalStatus::BudgetLimited
                && !budget_limit_was_already_reported;
        let goal_status = goal.status;
        let goal_id = goal.goal_id.clone();
        if goal_status != codex_state::ThreadGoalStatus::BudgetLimited {
            *self.goal_runtime.budget_limit_reported_goal_id.lock().await = None;
        }
        let goal = protocol_goal_from_state(goal);
        self.send_event(
            turn_context,
            EventMsg::ThreadGoalUpdated(ThreadGoalUpdatedEvent {
                thread_id: self.thread_id,
                turn_id: Some(turn_context.sub_id.clone()),
                goal: goal.clone(),
            }),
        )
        .await;
        if should_steer_budget_limit {
            let item = budget_limit_steering_item(&goal);
            if self.inject_if_running(vec![item]).await.is_err() {
                tracing::debug!("skipping budget-limit goal steering because no turn is active");
            }
            *self.goal_runtime.budget_limit_reported_goal_id.lock().await = Some(goal_id);
        }
        Ok(())
    }

    async fn account_thread_goal_before_external_mutation(&self) -> anyhow::Result<()> {
        if let Some(turn_context) = self.active_turn_context().await {
            return self
                .account_thread_goal_progress(
                    turn_context.as_ref(),
                    BudgetLimitSteering::Suppressed,
                    TerminalMetricEmission::Emit,
                )
                .await;
        }

        let Some(state_db) = self.state_db_for_thread_goals().await? else {
            return Ok(());
        };
        self.account_thread_goal_wall_clock_usage(
            &state_db,
            codex_state::GoalAccountingMode::ActiveOnly,
            TerminalMetricEmission::Suppress,
        )
        .await?;
        Ok(())
    }

    async fn account_thread_goal_wall_clock_usage(
        &self,
        state_db: &StateDbHandle,
        mode: codex_state::GoalAccountingMode,
        terminal_metric_emission: TerminalMetricEmission,
    ) -> anyhow::Result<Option<ThreadGoal>> {
        let _accounting_permit = self.goal_runtime.accounting_permit().await?;
        let (time_delta_seconds, expected_goal_id) = {
            let accounting = self.goal_runtime.accounting.lock().await;
            (
                accounting.wall_clock.time_delta_since_last_accounting(),
                accounting.wall_clock.active_goal_id(),
            )
        };
        if time_delta_seconds == 0 {
            return Ok(None);
        }
        let previous_status = self
            .current_goal_status_for_metrics(state_db, expected_goal_id.as_deref())
            .await?;

        match state_db
            .thread_goals()
            .account_thread_goal_usage(
                self.thread_id,
                time_delta_seconds,
                /*token_delta*/ 0,
                mode,
                expected_goal_id.as_deref(),
            )
            .await?
        {
            codex_state::GoalAccountingOutcome::Updated(goal) => {
                if matches!(terminal_metric_emission, TerminalMetricEmission::Emit) {
                    self.emit_goal_terminal_metrics_if_status_changed(previous_status, &goal);
                }
                self.goal_runtime
                    .accounting
                    .lock()
                    .await
                    .wall_clock
                    .mark_accounted(time_delta_seconds);
                let goal = protocol_goal_from_state(goal);
                Ok(Some(goal))
            }
            codex_state::GoalAccountingOutcome::Unchanged(goal) => {
                {
                    let mut accounting = self.goal_runtime.accounting.lock().await;
                    accounting.wall_clock.reset_baseline();
                    accounting.wall_clock.clear_active_goal();
                }
                if let Some(goal) = goal {
                    let goal = protocol_goal_from_state(goal);
                    return Ok(Some(goal));
                }
                Ok(None)
            }
        }
    }

    async fn usage_limit_active_thread_goal_for_turn(
        &self,
        turn_context: &TurnContext,
    ) -> anyhow::Result<()> {
        if should_ignore_goal_for_mode(turn_context.collaboration_mode.mode) {
            return Ok(());
        }

        if !self.enabled(Feature::Goals) {
            return Ok(());
        }

        let _continuation_guard = self
            .goal_runtime
            .continuation_lock
            .acquire()
            .await
            .context("goal continuation semaphore closed")?;
        let Some(state_db) = self.state_db_for_thread_goals().await? else {
            return Ok(());
        };
        self.account_thread_goal_progress(
            turn_context,
            BudgetLimitSteering::Suppressed,
            TerminalMetricEmission::Emit,
        )
        .await?;
        let previous_status = self
            .current_goal_status_for_metrics(&state_db, /*expected_goal_id*/ None)
            .await?;
        let Some(goal) = state_db
            .thread_goals()
            .usage_limit_active_thread_goal(self.thread_id)
            .await?
        else {
            return Ok(());
        };
        self.emit_goal_terminal_metrics_if_status_changed(previous_status, &goal);
        let goal = protocol_goal_from_state(goal);
        *self.goal_runtime.budget_limit_reported_goal_id.lock().await = None;
        self.clear_active_goal_accounting(turn_context).await;
        self.send_event(
            turn_context,
            EventMsg::ThreadGoalUpdated(ThreadGoalUpdatedEvent {
                thread_id: self.thread_id,
                turn_id: Some(turn_context.sub_id.clone()),
                goal,
            }),
        )
        .await;
        Ok(())
    }

    async fn restore_thread_goal_runtime_after_resume(&self) -> anyhow::Result<()> {
        if !self.enabled(Feature::Goals) {
            return Ok(());
        }
        if should_ignore_goal_for_mode(self.collaboration_mode().await.mode) {
            tracing::debug!(
                "skipping goal runtime restore while current collaboration mode ignores goals"
            );
            return Ok(());
        }

        let _continuation_guard = self
            .goal_runtime
            .continuation_lock
            .acquire()
            .await
            .context("goal continuation semaphore closed")?;
        let Some(state_db) = self.state_db_for_thread_goals().await? else {
            return Ok(());
        };
        let Some(goal) = state_db
            .thread_goals()
            .get_thread_goal(self.thread_id)
            .await?
        else {
            self.clear_stopped_thread_goal_runtime_state().await;
            return Ok(());
        };
        match goal.status {
            codex_state::ThreadGoalStatus::Active => {
                self.goal_runtime
                    .accounting
                    .lock()
                    .await
                    .wall_clock
                    .mark_active_goal(goal.goal_id);
                self.emit_goal_resumed_metric();
            }
            codex_state::ThreadGoalStatus::Paused
            | codex_state::ThreadGoalStatus::Blocked
            | codex_state::ThreadGoalStatus::UsageLimited
            | codex_state::ThreadGoalStatus::BudgetLimited
            | codex_state::ThreadGoalStatus::Complete => {
                self.clear_stopped_thread_goal_runtime_state().await;
            }
        }
        Ok(())
    }

    async fn maybe_continue_goal_if_idle_runtime(self: &Arc<Self>) {
        self.maybe_start_turn_for_pending_work().await;
        self.maybe_start_goal_continuation_turn().await;
    }

    async fn maybe_start_goal_continuation_turn(self: &Arc<Self>) {
        let Ok(_continuation_guard) = self.goal_runtime.continuation_lock.acquire().await else {
            tracing::warn!("goal continuation semaphore closed");
            return;
        };
        let Some(candidate) = self.goal_continuation_candidate_if_active().await else {
            return;
        };

        let turn_state = {
            let mut active_turn = self.active_turn.lock().await;
            if active_turn.is_some() {
                return;
            }
            let active_turn = active_turn.get_or_insert_with(ActiveTurn::default);
            Arc::clone(&active_turn.turn_state)
        };
        let goal_is_current = match self.state_db_for_thread_goals().await {
            Ok(Some(state_db)) => match state_db
                .thread_goals()
                .get_thread_goal(self.thread_id)
                .await
            {
                Ok(Some(goal))
                    if goal.goal_id == candidate.goal_id
                        && goal.status == codex_state::ThreadGoalStatus::Active =>
                {
                    true
                }
                Ok(Some(_)) | Ok(None) => {
                    tracing::debug!(
                        "skipping active goal continuation because the goal changed before launch"
                    );
                    false
                }
                Err(err) => {
                    tracing::warn!("failed to re-read thread goal before continuation: {err}");
                    false
                }
            },
            Ok(None) => {
                tracing::debug!("skipping active goal continuation for ephemeral thread");
                false
            }
            Err(err) => {
                tracing::warn!("failed to open state db before goal continuation: {err}");
                false
            }
        };
        if !goal_is_current {
            self.clear_reserved_goal_continuation_turn(&turn_state)
                .await;
            return;
        }
        self.input_queue
            .extend_pending_input_for_turn_state(
                turn_state.as_ref(),
                candidate
                    .items
                    .into_iter()
                    .map(TurnInput::ResponseItem)
                    .collect(),
            )
            .await;

        let turn_context = self
            .new_default_turn_with_sub_id(uuid::Uuid::new_v4().to_string())
            .await;
        self.maybe_emit_unknown_model_warning_for_turn(turn_context.as_ref())
            .await;
        let still_reserved = {
            let active_turn = self.active_turn.lock().await;
            active_turn.as_ref().is_some_and(|active_turn| {
                active_turn.task.is_none() && Arc::ptr_eq(&active_turn.turn_state, &turn_state)
            })
        };
        if !still_reserved {
            self.clear_reserved_goal_continuation_turn(&turn_state)
                .await;
            return;
        }
        self.start_task(turn_context, Vec::new(), RegularTask::new())
            .await;
    }

    async fn goal_continuation_candidate_if_active(
        self: &Arc<Self>,
    ) -> Option<GoalContinuationCandidate> {
        if !self.enabled(Feature::Goals) {
            return None;
        }
        if should_ignore_goal_for_mode(self.collaboration_mode().await.mode) {
            tracing::debug!("skipping active goal continuation while plan mode is active");
            return None;
        }
        if self.active_turn.lock().await.is_some() {
            tracing::debug!("skipping active goal continuation because a turn is already active");
            return None;
        }
        if self.input_queue.has_trigger_turn_mailbox_items().await {
            tracing::debug!(
                "skipping active goal continuation because trigger-turn mailbox input is pending"
            );
            return None;
        }
        let state_db = match self.state_db_for_thread_goals().await {
            Ok(Some(state_db)) => state_db,
            Ok(None) => {
                tracing::debug!("skipping active goal continuation for ephemeral thread");
                return None;
            }
            Err(err) => {
                tracing::warn!("failed to open state db for goal continuation: {err}");
                return None;
            }
        };
        let goal = match state_db
            .thread_goals()
            .get_thread_goal(self.thread_id)
            .await
        {
            Ok(Some(goal)) => goal,
            Ok(None) => {
                tracing::debug!("skipping active goal continuation because no goal is set");
                return None;
            }
            Err(err) => {
                tracing::warn!("failed to read thread goal for continuation: {err}");
                return None;
            }
        };
        if goal.status != codex_state::ThreadGoalStatus::Active {
            tracing::debug!(status = ?goal.status, "skipping inactive thread goal");
            return None;
        }
        if self.active_turn.lock().await.is_some()
            || self.input_queue.has_trigger_turn_mailbox_items().await
        {
            tracing::debug!("skipping active goal continuation because pending work appeared");
            return None;
        }
        let goal_id = goal.goal_id.clone();
        let goal = protocol_goal_from_state(goal);
        // For the auto-armed Gemini orchestration goal, inject the
        // orchestration continuation prompt (treat children's reports as
        // authoritative; don't re-investigate/re-plan delegated work) instead
        // of the solo-worker prompt. The marker is only ever set on a
        // Gemini-native root auto-arm, so non-Gemini/user goals are byte-identical.
        let is_auto_armed_orchestration_goal = *self
            .goal_runtime
            .auto_armed_orchestration_goal_id
            .lock()
            .await
            == Some(goal_id.clone());
        let prompt = if is_auto_armed_orchestration_goal {
            orchestration_continuation_prompt(&goal)
        } else {
            continuation_prompt(&goal)
        };
        Some(GoalContinuationCandidate {
            goal_id,
            items: vec![goal_context_input_item(prompt)],
        })
    }

    /// Structural termination for a Gemini-native root orchestrator: once the
    /// auto-armed goal's delegated work is all terminal-with-delivery and a
    /// consolidated answer has been emitted (this turn or a prior turn since the
    /// last child re-engagement), complete the goal so the idle continuation
    /// backstop stops re-engaging the (now-done) orchestrator. This replaces the
    /// dependence on the model remembering to call `update_goal complete`.
    ///
    /// Called from `on_task_finished` immediately before `MaybeContinueIfIdle`,
    /// on a turn that already cleared the active turn (i.e. the session is
    /// idle). Completing here makes the very next continuation candidate
    /// short-circuit on `status != Active`.
    ///
    /// O27: the "answer emitted" requirement is satisfied durably across turns
    /// via `orchestration_answer_emitted_goal_id`, so the common
    /// synthesize-then-coordinate sequence (emit the consolidated answer, then a
    /// text-free idle continuation turn) still completes — the prior dependence
    /// on the answer landing on the *same* idle turn caused the budget-burning
    /// re-engagement grind.
    ///
    /// Anti-stall (O15): whenever any guard fails we leave the goal `Active`, so
    /// the pre-existing backstop is untouched. We only complete when no child was
    /// re-engaged this turn, an answer was emitted since the last re-engagement,
    /// nothing is pending in the trigger-turn mailbox, and every enumerable child
    /// is terminal-with-delivery. A `Completed(None)` (null) child is *not*
    /// terminal-with-delivery, so the parent keeps chasing a real report for it.
    pub(crate) async fn maybe_auto_complete_gemini_orchestration_goal(
        &self,
        turn_context: &TurnContext,
        emitted_final_answer: bool,
        reengaged_child_this_turn: bool,
    ) {
        // G0: a turn that re-engaged a child is never a completion turn, and the
        // re-engagement already cleared the durable answer marker in
        // `mark_reengaged_child_this_turn`. Cheapest gate first — this runs on
        // every idle turn-end for all sessions, so bail before any lock/db work.
        if reengaged_child_this_turn {
            return;
        }
        // G1: Gemini-native root orchestrator only. Non-Gemini / non-root is a
        // byte-identical no-op (returns here exactly as before O27).
        if turn_context.provider.info().wire_api != WireApi::GeminiNative
            || !is_root_orchestrator_source(&turn_context.session_source)
        {
            return;
        }
        // G2: only an auto-armed orchestration goal is ever eligible.
        let marker_goal_id = {
            let marker = self
                .goal_runtime
                .auto_armed_orchestration_goal_id
                .lock()
                .await;
            match marker.as_ref() {
                Some(goal_id) => goal_id.clone(),
                None => return,
            }
        };
        // G3: record the durable answer marker for this goal when this turn
        // emitted a final-channel answer (and, per G0, did not re-engage a
        // child). Keyed by exact goal id so a stale value never matches a
        // different/replaced goal.
        if emitted_final_answer {
            *self
                .goal_runtime
                .orchestration_answer_emitted_goal_id
                .lock()
                .await = Some(marker_goal_id.clone());
        }
        // G4: a consolidated answer must have been emitted at least once since
        // the last re-engagement — this turn (G3) or a durable prior turn.
        let answered = *self
            .goal_runtime
            .orchestration_answer_emitted_goal_id
            .lock()
            .await
            == Some(marker_goal_id.clone());
        if !answered {
            return;
        }
        // G5: an uncollected non-empty child completion would wake the parent via
        // the trigger-turn mailbox; don't complete while one is pending.
        if self.input_queue.has_trigger_turn_mailbox_items().await {
            return;
        }
        // G6: the goal must still be the exact auto-armed goal and still Active.
        let goal = match self.state_db_for_thread_goals().await {
            Ok(Some(state_db)) => match state_db
                .thread_goals()
                .get_thread_goal(self.thread_id)
                .await
            {
                Ok(goal) => goal,
                Err(err) => {
                    tracing::warn!(
                        "skipping orchestration auto-complete: failed to read thread goal: {err}"
                    );
                    return;
                }
            },
            Ok(None) => return,
            Err(err) => {
                tracing::warn!(
                    "skipping orchestration auto-complete: failed to open state db: {err}"
                );
                return;
            }
        };
        let Some(goal) = goal else {
            return;
        };
        if goal.status != codex_state::ThreadGoalStatus::Active || goal.goal_id != marker_goal_id {
            return;
        }
        // G7: every delegated child must be terminal-with-delivery. An empty list
        // is vacuously done (e.g. all children explicitly closed). A `Completed(None)`
        // (null) child or any non-terminal child (NotFound/PendingInit/Running/
        // Interrupted) keeps the goal Active so the continuation re-wakes the
        // parent to chase a real report.
        let children = match self
            .services
            .agent_control
            .open_thread_spawn_children(self.thread_id)
            .await
        {
            Ok(children) => children,
            Err(err) => {
                tracing::warn!(
                    "skipping orchestration auto-complete: failed to enumerate children: {err}"
                );
                return;
            }
        };
        for (child_thread_id, _metadata) in &children {
            let status: AgentStatus = self
                .services
                .agent_control
                .get_status(*child_thread_id)
                .await;
            if !is_terminal_with_delivery(&status) {
                return;
            }
        }

        // Genuinely done: complete the goal exactly as the model's
        // `update_goal complete` would. `set_thread_goal` performs final
        // wall-clock accounting, flips Active->Complete, and emits
        // GOAL_COMPLETED. (No `ToolCompletedGoal` event: the prior
        // `TurnFinished` already cleared the per-turn accounting snapshot, so
        // it would be a no-op here.)
        if let Err(err) = self
            .set_thread_goal(
                turn_context,
                SetGoalRequest {
                    objective: None,
                    status: Some(ThreadGoalStatus::Complete),
                    token_budget: None,
                },
            )
            .await
        {
            tracing::warn!("failed to auto-complete Gemini orchestration goal: {err}");
        } else {
            tracing::debug!(
                "auto-completed Gemini orchestration goal {marker_goal_id} on clean synthesis turn"
            );
        }
    }
}

impl Session {
    async fn state_db_for_thread_goals(&self) -> anyhow::Result<Option<StateDbHandle>> {
        let config = self.get_config().await;
        if config.ephemeral {
            return Ok(None);
        }

        self.try_ensure_rollout_materialized()
            .await
            .context("failed to materialize rollout before opening state db for thread goals")?;

        let state_db = if let Some(state_db) = self.state_db() {
            state_db
        } else if let Some(state_db) = self.goal_runtime.state_db.lock().await.clone() {
            state_db
        } else if let Some(local_store) = self
            .services
            .thread_store
            .as_any()
            .downcast_ref::<LocalThreadStore>()
        {
            local_store.state_db().await.ok_or_else(|| {
                anyhow::anyhow!(
                    "thread goals require a local persisted thread with a state database"
                )
            })?
        } else {
            anyhow::bail!("thread goals require a local persisted thread with a state database");
        };

        let thread_metadata_present = state_db
            .get_thread(self.thread_id)
            .await
            .context("failed to read thread metadata before reconciling thread goals")?
            .is_some();
        if !thread_metadata_present {
            let rollout_path = self
                .current_rollout_path()
                .await
                .context("failed to locate rollout before reconciling thread goals")?
                .ok_or_else(|| {
                    anyhow::anyhow!("thread goals require materialized thread metadata")
                })?;
            reconcile_rollout(
                Some(&state_db),
                rollout_path.as_path(),
                config.model_provider_id.as_str(),
                /*builder*/ None,
                &[],
                /*archived_only*/ None,
                /*new_thread_memory_mode*/ None,
            )
            .await;
            let thread_metadata_present = state_db
                .get_thread(self.thread_id)
                .await
                .context("failed to read thread metadata after reconciling thread goals")?
                .is_some();
            if !thread_metadata_present {
                anyhow::bail!("thread metadata is unavailable after reconciling thread goals");
            }
        }

        *self.goal_runtime.state_db.lock().await = Some(state_db.clone());
        Ok(Some(state_db))
    }

    async fn require_state_db_for_thread_goals(&self) -> anyhow::Result<StateDbHandle> {
        self.state_db_for_thread_goals().await?.ok_or_else(|| {
            anyhow::anyhow!("thread goals require a persisted thread; this thread is ephemeral")
        })
    }
}

async fn set_thread_preview_from_goal_objective(
    state_db: &StateDbHandle,
    thread_id: ThreadId,
    objective: &str,
) {
    if let Err(err) = state_db
        .set_thread_preview_if_empty(thread_id, objective)
        .await
    {
        tracing::warn!(
            "failed to set empty thread preview from goal objective for {thread_id}: {err}"
        );
    }
}

fn should_ignore_goal_for_mode(mode: ModeKind) -> bool {
    mode == ModeKind::Plan
}

fn budget_limit_steering_item(goal: &ThreadGoal) -> ResponseItem {
    goal_context_input_item(budget_limit_prompt(goal))
}

fn goal_context_input_item(prompt: String) -> ResponseItem {
    ContextualUserFragment::into(InternalModelContextFragment::new(
        InternalContextSource::from_static("goal"),
        prompt,
    ))
}

pub(crate) fn protocol_goal_from_state(goal: codex_state::ThreadGoal) -> ThreadGoal {
    ThreadGoal {
        thread_id: goal.thread_id,
        objective: goal.objective,
        status: protocol_goal_status_from_state(goal.status),
        token_budget: goal.token_budget,
        tokens_used: goal.tokens_used,
        time_used_seconds: goal.time_used_seconds,
        created_at: goal.created_at.timestamp(),
        updated_at: goal.updated_at.timestamp(),
    }
}

pub(crate) fn protocol_goal_status_from_state(
    status: codex_state::ThreadGoalStatus,
) -> ThreadGoalStatus {
    match status {
        codex_state::ThreadGoalStatus::Active => ThreadGoalStatus::Active,
        codex_state::ThreadGoalStatus::Paused => ThreadGoalStatus::Paused,
        codex_state::ThreadGoalStatus::Blocked => ThreadGoalStatus::Blocked,
        codex_state::ThreadGoalStatus::UsageLimited => ThreadGoalStatus::UsageLimited,
        codex_state::ThreadGoalStatus::BudgetLimited => ThreadGoalStatus::BudgetLimited,
        codex_state::ThreadGoalStatus::Complete => ThreadGoalStatus::Complete,
    }
}

pub(crate) fn state_goal_status_from_protocol(
    status: ThreadGoalStatus,
) -> codex_state::ThreadGoalStatus {
    match status {
        ThreadGoalStatus::Active => codex_state::ThreadGoalStatus::Active,
        ThreadGoalStatus::Paused => codex_state::ThreadGoalStatus::Paused,
        ThreadGoalStatus::Blocked => codex_state::ThreadGoalStatus::Blocked,
        ThreadGoalStatus::UsageLimited => codex_state::ThreadGoalStatus::UsageLimited,
        ThreadGoalStatus::BudgetLimited => codex_state::ThreadGoalStatus::BudgetLimited,
        ThreadGoalStatus::Complete => codex_state::ThreadGoalStatus::Complete,
    }
}

pub(crate) fn validate_goal_budget(value: Option<i64>) -> anyhow::Result<()> {
    if let Some(value) = value
        && value <= 0
    {
        anyhow::bail!("goal budgets must be positive when provided");
    }
    Ok(())
}

pub(crate) fn goal_token_delta_for_usage(usage: &TokenUsage) -> i64 {
    usage
        .non_cached_input()
        .saturating_add(usage.output_tokens.max(0))
}

#[cfg(test)]
mod tests {
    use super::goal_context_input_item;
    use super::goal_token_delta_for_usage;
    use super::should_ignore_goal_for_mode;
    use codex_protocol::config_types::ModeKind;
    use codex_protocol::models::ContentItem;
    use codex_protocol::models::ResponseItem;
    use codex_protocol::protocol::TokenUsage;
    use std::time::Duration;
    use std::time::Instant;

    #[test]
    fn goal_continuation_is_ignored_only_in_plan_mode() {
        assert!(should_ignore_goal_for_mode(ModeKind::Plan));
        assert!(!should_ignore_goal_for_mode(ModeKind::Default));
        assert!(!should_ignore_goal_for_mode(ModeKind::PairProgramming));
        assert!(!should_ignore_goal_for_mode(ModeKind::Execute));
    }

    #[test]
    fn goal_token_delta_excludes_cached_input_and_does_not_double_count_reasoning() {
        let usage = TokenUsage {
            input_tokens: 900,
            cached_input_tokens: 400,
            output_tokens: 80,
            reasoning_output_tokens: 20,
            total_tokens: 1_000,
        };

        assert_eq!(580, goal_token_delta_for_usage(&usage));
    }

    #[test]
    fn wall_clock_accounting_advances_by_persisted_seconds() {
        let mut snapshot = super::GoalWallClockAccountingSnapshot::new();
        let original = Instant::now() - Duration::from_millis(1500);
        snapshot.last_accounted_at = original;

        snapshot.mark_accounted(/*accounted_seconds*/ 1);
        assert_eq!(
            original + Duration::from_secs(1),
            snapshot.last_accounted_at
        );

        let token_only_original = snapshot.last_accounted_at;
        snapshot.mark_accounted(/*accounted_seconds*/ 0);
        assert_eq!(token_only_original, snapshot.last_accounted_at);
    }

    #[test]
    fn goal_context_input_item_is_hidden_user_context() {
        let item = goal_context_input_item("Continue working.".to_string());

        assert_eq!(
            item,
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "<codex_internal_context source=\"goal\">\nContinue working.\n</codex_internal_context>".to_string(),
                }],
                phase: None,
            }
        );
    }
}
