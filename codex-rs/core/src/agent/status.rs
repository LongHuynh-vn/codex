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

/// Stricter than [`is_final`]: a child is only "done" for orchestration
/// auto-completion purposes when it reached a genuinely terminal, settled
/// state. Unlike `is_final`, this treats `NotFound` as **not** done so a
/// transiently-unregistered or mid-spawn child can never be mistaken for a
/// finished one. The not-done set is exactly
/// `{NotFound, PendingInit, Running, Interrupted}`.
///
/// `Shutdown` is matched for safety/future-proofing, but is currently
/// unreachable as an *observed* status: shutting a child down removes it from
/// `open_thread_spawn_children`, so shut-down children are handled by the
/// empty-list (vacuously-terminal) case at the call site instead. The
/// reachable terminal states for enumerable children are `Completed`/`Errored`.
pub(crate) fn is_conservatively_terminal(status: &AgentStatus) -> bool {
    matches!(
        status,
        AgentStatus::Completed(_) | AgentStatus::Errored(_) | AgentStatus::Shutdown
    )
}
