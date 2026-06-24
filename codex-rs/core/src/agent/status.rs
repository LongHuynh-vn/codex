use codex_protocol::protocol::AgentStatus;
use codex_protocol::protocol::EventMsg;

/// Derive the next agent status from a single emitted event.
/// Returns `None` when the event does not affect status tracking.
pub(crate) fn agent_status_from_event(msg: &EventMsg) -> Option<AgentStatus> {
    match msg {
        EventMsg::TurnStarted(_) => Some(AgentStatus::Running),
        EventMsg::TurnComplete(ev) => Some(AgentStatus::Completed(ev.last_agent_message.clone())),
        EventMsg::TurnAborted(ev) => match ev.reason {
            codex_protocol::protocol::TurnAbortReason::Interrupted
            | codex_protocol::protocol::TurnAbortReason::BudgetLimited => {
                Some(AgentStatus::Interrupted)
            }
            _ => Some(AgentStatus::Errored(format!("{:?}", ev.reason))),
        },
        EventMsg::Error(ev) => Some(AgentStatus::Errored(ev.message.clone())),
        EventMsg::ShutdownComplete => Some(AgentStatus::Shutdown),
        _ => None,
    }
}

pub(crate) fn is_final(status: &AgentStatus) -> bool {
    !matches!(
        status,
        AgentStatus::PendingInit | AgentStatus::Running | AgentStatus::Interrupted
    )
}

/// Stricter than [`is_final`]: a child counts as "done with a delivered result"
/// for orchestration auto-completion only when it reached a genuinely terminal,
/// settled state that carried something the orchestrator can synthesize from — a
/// report (`Completed(Some(..))`) or a determinate error/shutdown verdict.
///
/// Unlike `is_final`, this treats `NotFound` as **not** done so a transiently-
/// unregistered or mid-spawn child is never mistaken for a finished one; the
/// non-terminal not-done set is `{NotFound, PendingInit, Running, Interrupted}`
/// (a child running its child-side empty-report retry is `Running`, so it is
/// never counted as done while a report could still arrive).
///
/// `Completed(None)` — the degenerate empty/null-report case that O24/O27 drives
/// toward zero — is also **not** done: it is the one *indeterminate* terminal
/// outcome, delivering no report and recoverable (the child-side bounded retry
/// recovers it most of the time, and the parent can `followup_task` it). Keeping
/// it not-done leaves the orchestration goal Active so the parent chases a real
/// report instead of finalizing with a missing section. `Errored` IS delivered:
/// an error is a determinate terminal verdict, and re-engaging a
/// persistently-errored child is the re-engagement spin O27 exists to remove —
/// the orchestrator must surface the error in its synthesis instead.
///
/// `Shutdown` is matched for safety/future-proofing, but is currently
/// unreachable as an *observed* status: shutting a child down removes it from
/// `open_thread_spawn_children`, so shut-down children are handled by the
/// empty-list (vacuously-done) case at the call site instead.
pub(crate) fn is_terminal_with_delivery(status: &AgentStatus) -> bool {
    matches!(
        status,
        AgentStatus::Completed(Some(_)) | AgentStatus::Errored(_) | AgentStatus::Shutdown
    )
}
