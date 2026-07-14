//! Session-owned token-budget accounting for Gemini-native spawned children.
//!
//! Runtime gating remains at the turn-loop callsite. This module only owns the
//! fixed per-session baseline, one-shot reminder latch, and exhaustion-grace
//! phase once that gated callsite initializes it.

use crate::goals::goal_token_delta_for_usage;
use codex_protocol::protocol::TokenUsage;
use tokio::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChildTokenBudgetAction {
    None,
    Reminder {
        spent: i64,
        remaining: i64,
        limit: i64,
    },
    BeginExhaustionGrace {
        spent: i64,
        limit: i64,
    },
    ForceCompletion {
        spent: i64,
        limit: i64,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ExhaustionPhase {
    #[default]
    Normal,
    GraceInFlight,
    ForceCompletion,
}

#[derive(Debug, Default)]
struct ChildTokenBudgetRuntimeInner {
    baseline: Option<TokenUsage>,
    limit: i64,
    reminder_fired: bool,
    exhaustion_phase: ExhaustionPhase,
}

/// In-memory token-budget state fixed to a spawned child's session lifetime.
pub(crate) struct ChildTokenBudgetRuntimeState {
    inner: Mutex<ChildTokenBudgetRuntimeInner>,
}

impl ChildTokenBudgetRuntimeState {
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(ChildTokenBudgetRuntimeInner::default()),
        }
    }

    /// Captures the inherited-usage baseline and configured limit once.
    pub(crate) async fn initialize(&self, baseline: TokenUsage, limit: i64) {
        let mut inner = self.inner.lock().await;
        if inner.baseline.is_some() {
            return;
        }
        inner.baseline = Some(baseline);
        inner.limit = limit.max(0);
    }

    /// Observes cumulative session usage after a completed sampling request.
    pub(crate) async fn observe(&self, current: &TokenUsage) -> ChildTokenBudgetAction {
        let mut inner = self.inner.lock().await;
        let Some(baseline) = inner.baseline.as_ref() else {
            return ChildTokenBudgetAction::None;
        };
        if inner.limit == 0 {
            return ChildTokenBudgetAction::None;
        }

        let spent = child_token_spend(&token_usage_delta(current, baseline));
        let limit = inner.limit;

        match inner.exhaustion_phase {
            ExhaustionPhase::GraceInFlight => {
                inner.exhaustion_phase = ExhaustionPhase::ForceCompletion;
                return ChildTokenBudgetAction::ForceCompletion { spent, limit };
            }
            ExhaustionPhase::ForceCompletion => {
                return ChildTokenBudgetAction::ForceCompletion { spent, limit };
            }
            ExhaustionPhase::Normal => {}
        }

        if spent >= limit {
            inner.exhaustion_phase = ExhaustionPhase::GraceInFlight;
            return ChildTokenBudgetAction::BeginExhaustionGrace { spent, limit };
        }

        let reminder_threshold = limit.saturating_sub(limit / 5);
        if !inner.reminder_fired && spent >= reminder_threshold {
            inner.reminder_fired = true;
            return ChildTokenBudgetAction::Reminder {
                spent,
                remaining: limit.saturating_sub(spent).max(0),
                limit,
            };
        }

        ChildTokenBudgetAction::None
    }

    pub(crate) async fn exhaustion_grace_in_flight(&self) -> bool {
        self.inner.lock().await.exhaustion_phase == ExhaustionPhase::GraceInFlight
    }

    pub(crate) async fn has_force_completed(&self) -> bool {
        self.inner.lock().await.exhaustion_phase == ExhaustionPhase::ForceCompletion
    }
}

fn token_usage_delta(current: &TokenUsage, baseline: &TokenUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: current
            .input_tokens
            .saturating_sub(baseline.input_tokens)
            .max(0),
        cached_input_tokens: current
            .cached_input_tokens
            .saturating_sub(baseline.cached_input_tokens)
            .max(0),
        output_tokens: current
            .output_tokens
            .saturating_sub(baseline.output_tokens)
            .max(0),
        reasoning_output_tokens: current
            .reasoning_output_tokens
            .saturating_sub(baseline.reasoning_output_tokens)
            .max(0),
        total_tokens: current
            .total_tokens
            .saturating_sub(baseline.total_tokens)
            .max(0),
    }
}

fn child_token_spend(usage: &TokenUsage) -> i64 {
    goal_token_delta_for_usage(usage).saturating_add(usage.reasoning_output_tokens.max(0))
}

#[cfg(test)]
#[path = "child_token_budget_tests.rs"]
mod tests;
