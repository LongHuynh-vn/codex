use super::*;
use crate::agent::status::is_final;
use crate::config::MultiAgentV2Config;
use crate::tools::handlers::multi_agents_spec::WaitAgentTimeoutOptions;
use crate::tools::handlers::multi_agents_spec::WaitAgentV2OutputMode;
use crate::tools::handlers::multi_agents_spec::create_wait_agent_tool_v2;
use crate::turn_timing::now_unix_timestamp_ms;
use codex_model_provider_info::WireApi;
use codex_protocol::ThreadId;
use codex_protocol::error::CodexErr;
use codex_protocol::protocol::CollabAgentRef;
use codex_protocol::protocol::CollabAgentStatusEntry;
use codex_tools::ToolSpec;
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch::Receiver;
use tokio::time::Instant;
use tokio::time::sleep_until;
use tokio::time::timeout_at;

pub(crate) const GEMINI_DEFAULT_MULTI_AGENT_V2_WAIT_TIMEOUT_MS: i64 = 120_000;

pub(crate) fn effective_wait_agent_v2_timeout_options(
    config: &MultiAgentV2Config,
    wire_api: WireApi,
) -> WaitAgentTimeoutOptions {
    let default_timeout_ms = if wire_api == WireApi::GeminiNative {
        GEMINI_DEFAULT_MULTI_AGENT_V2_WAIT_TIMEOUT_MS
            .clamp(config.min_wait_timeout_ms, config.max_wait_timeout_ms)
    } else {
        config.default_wait_timeout_ms
    };

    WaitAgentTimeoutOptions {
        default_timeout_ms,
        min_timeout_ms: config.min_wait_timeout_ms,
        max_timeout_ms: config.max_wait_timeout_ms,
    }
}

#[derive(Default)]
pub(crate) struct Handler {
    options: WaitAgentTimeoutOptions,
    output_mode: WaitAgentV2OutputMode,
}

impl Handler {
    pub(crate) fn new_with_output_mode(
        options: WaitAgentTimeoutOptions,
        output_mode: WaitAgentV2OutputMode,
    ) -> Self {
        Self {
            options,
            output_mode,
        }
    }
}

#[async_trait::async_trait]
impl ToolExecutor<ToolInvocation> for Handler {
    fn tool_name(&self) -> ToolName {
        ToolName::plain("wait_agent")
    }

    fn spec(&self) -> ToolSpec {
        create_wait_agent_tool_v2(self.options, self.output_mode)
    }

    async fn handle(
        &self,
        invocation: ToolInvocation,
    ) -> Result<Box<dyn crate::tools::context::ToolOutput>, FunctionCallError> {
        let ToolInvocation {
            session,
            turn,
            payload,
            call_id,
            ..
        } = invocation;
        let arguments = function_arguments(payload)?;
        let args: WaitArgs = parse_arguments(&arguments)?;
        let options = effective_wait_agent_v2_timeout_options(
            &turn.config.multi_agent_v2,
            turn.provider.info().wire_api,
        );
        let min_timeout_ms = options.min_timeout_ms;
        let max_timeout_ms = options.max_timeout_ms;
        let default_timeout_ms = options.default_timeout_ms;
        let timeout_ms = match args.timeout_ms {
            Some(ms) if ms < min_timeout_ms => {
                return Err(FunctionCallError::RespondToModel(format!(
                    "timeout_ms must be at least {min_timeout_ms}"
                )));
            }
            Some(ms) if ms > max_timeout_ms => {
                return Err(FunctionCallError::RespondToModel(format!(
                    "timeout_ms must be at most {max_timeout_ms}"
                )));
            }
            Some(ms) => ms,
            None => default_timeout_ms,
        };

        let mut mailbox_rx = session.input_queue.subscribe_mailbox().await;

        session
            .send_event(
                &turn,
                CollabWaitingBeginEvent {
                    started_at_ms: now_unix_timestamp_ms(),
                    sender_thread_id: session.thread_id,
                    receiver_thread_ids: Vec::new(),
                    receiver_agents: Vec::new(),
                    call_id: call_id.clone(),
                }
                .into(),
            )
            .await;

        let deadline = Instant::now() + Duration::from_millis(timeout_ms as u64);
        let use_status_output = self.output_mode == WaitAgentV2OutputMode::GeminiStatuses
            && turn.provider.info().wire_api == WireApi::GeminiNative;
        if !use_status_output {
            let timed_out = !wait_for_mailbox_change(&mut mailbox_rx, deadline).await;
            let result = WaitAgentResult::from_timed_out(timed_out);

            session
                .send_event(
                    &turn,
                    CollabWaitingEndEvent {
                        sender_thread_id: session.thread_id,
                        call_id,
                        completed_at_ms: now_unix_timestamp_ms(),
                        agent_statuses: Vec::new(),
                        statuses: HashMap::new(),
                    }
                    .into(),
                )
                .await;

            return Ok(boxed_tool_output(result));
        }

        session
            .services
            .agent_control
            .register_session_root(session.thread_id, turn.parent_thread_id);
        let children = session
            .services
            .agent_control
            .open_thread_spawn_children(session.thread_id)
            .await
            .map_err(collab_spawn_error)?;
        let receiver_agents = children
            .into_iter()
            .map(|(thread_id, metadata)| CollabAgentRef {
                thread_id,
                agent_nickname: metadata.agent_nickname,
                agent_role: metadata.agent_role,
            })
            .collect::<Vec<_>>();
        if receiver_agents.is_empty() {
            let statuses = HashMap::new();
            let result = WaitAgentResult::with_statuses(
                "No live child agents to wait for.".to_string(),
                /*timed_out*/ false,
                statuses.clone(),
                &receiver_agents,
            );
            send_waiting_end_event(&session, &turn, call_id, statuses, &receiver_agents).await;
            return Ok(boxed_tool_output(result));
        }

        let subscriptions = match subscribe_child_statuses(&session, &receiver_agents).await {
            Ok(subscriptions) => subscriptions,
            Err(err) => {
                send_waiting_end_event(
                    &session,
                    &turn,
                    call_id.clone(),
                    err.statuses.clone(),
                    &receiver_agents,
                )
                .await;
                return Err(collab_agent_error(err.thread_id, err.err));
            }
        };
        let ChildStatusSubscriptions {
            status_rxs,
            terminal_statuses,
        } = subscriptions;

        // Block until every live child is terminal or the deadline elapses. We must NOT
        // return early just because some child was already terminal at entry, nor because
        // the parent mailbox fired (a child-completion notification leaves pending mail).
        let wait_outcome = wait_for_all_children_terminal(
            session.clone(),
            status_rxs,
            &terminal_statuses,
            &mut mailbox_rx,
            &receiver_agents,
            deadline,
        )
        .await;

        // Authoritative post-wait poll: the full terminal-status set to report.
        let statuses = final_statuses_for_children(&session, &receiver_agents).await;
        let all_terminal = receiver_agents
            .iter()
            .all(|agent| statuses.contains_key(&agent.thread_id));
        // timed_out only when the deadline fired and a live child is still non-terminal.
        let timed_out = matches!(wait_outcome, WaitOutcome::TimedOut) && !all_terminal;
        let result = WaitAgentResult::with_statuses(
            WaitAgentResult::message_for_timed_out(timed_out).to_string(),
            timed_out,
            statuses.clone(),
            &receiver_agents,
        );
        send_waiting_end_event(&session, &turn, call_id, statuses, &receiver_agents).await;
        Ok(boxed_tool_output(result))
    }
}

impl CoreToolRuntime for Handler {
    fn matches_kind(&self, payload: &ToolPayload) -> bool {
        matches!(payload, ToolPayload::Function { .. })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WaitArgs {
    timeout_ms: Option<i64>,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct WaitAgentResult {
    pub(crate) message: String,
    pub(crate) timed_out: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) statuses: Option<HashMap<ThreadId, AgentStatus>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) agent_statuses: Option<Vec<CollabAgentStatusEntry>>,
}

impl WaitAgentResult {
    fn from_timed_out(timed_out: bool) -> Self {
        Self {
            message: Self::message_for_timed_out(timed_out).to_string(),
            timed_out,
            statuses: None,
            agent_statuses: None,
        }
    }

    fn with_statuses(
        message: String,
        timed_out: bool,
        statuses: HashMap<ThreadId, AgentStatus>,
        receiver_agents: &[CollabAgentRef],
    ) -> Self {
        let agent_statuses = build_wait_agent_statuses(&statuses, receiver_agents);
        Self {
            message,
            timed_out,
            statuses: Some(statuses),
            agent_statuses: Some(agent_statuses),
        }
    }

    fn message_for_timed_out(timed_out: bool) -> &'static str {
        if timed_out {
            "Wait timed out."
        } else {
            "Wait completed."
        }
    }
}

impl ToolOutput for WaitAgentResult {
    fn log_preview(&self) -> String {
        tool_output_json_text(self, "wait_agent")
    }

    fn success_for_logging(&self) -> bool {
        true
    }

    fn to_response_item(&self, call_id: &str, payload: &ToolPayload) -> ResponseInputItem {
        tool_output_response_item(call_id, payload, self, /*success*/ None, "wait_agent")
    }

    fn code_mode_result(&self, _payload: &ToolPayload) -> JsonValue {
        tool_output_code_mode_result(self, "wait_agent")
    }
}

async fn send_waiting_end_event(
    session: &Arc<crate::session::session::Session>,
    turn: &Arc<crate::session::turn_context::TurnContext>,
    call_id: String,
    statuses: HashMap<ThreadId, AgentStatus>,
    receiver_agents: &[CollabAgentRef],
) {
    let agent_statuses = build_wait_agent_statuses(&statuses, receiver_agents);
    session
        .send_event(
            turn,
            CollabWaitingEndEvent {
                sender_thread_id: session.thread_id,
                call_id,
                completed_at_ms: now_unix_timestamp_ms(),
                agent_statuses,
                statuses,
            }
            .into(),
        )
        .await;
}

pub(crate) struct ChildStatusSubscriptions {
    pub(crate) status_rxs: Vec<(ThreadId, Receiver<AgentStatus>)>,
    pub(crate) terminal_statuses: HashMap<ThreadId, AgentStatus>,
}

#[derive(Debug)]
pub(crate) struct ChildStatusSubscribeError {
    thread_id: ThreadId,
    err: CodexErr,
    statuses: HashMap<ThreadId, AgentStatus>,
}

pub(crate) async fn subscribe_child_statuses(
    session: &crate::session::session::Session,
    receiver_agents: &[CollabAgentRef],
) -> Result<ChildStatusSubscriptions, ChildStatusSubscribeError> {
    let mut status_rxs = Vec::with_capacity(receiver_agents.len());
    let mut terminal_statuses = HashMap::new();
    for receiver_agent in receiver_agents {
        let thread_id = receiver_agent.thread_id;
        match session
            .services
            .agent_control
            .subscribe_status(thread_id)
            .await
        {
            Ok(rx) => {
                let status = rx.borrow().clone();
                if is_final(&status) {
                    terminal_statuses.insert(thread_id, status);
                }
                status_rxs.push((thread_id, rx));
            }
            Err(CodexErr::ThreadNotFound(_)) => {
                terminal_statuses.insert(thread_id, AgentStatus::NotFound);
            }
            Err(err) => {
                let mut statuses = HashMap::with_capacity(1);
                statuses.insert(
                    thread_id,
                    session.services.agent_control.get_status(thread_id).await,
                );
                return Err(ChildStatusSubscribeError {
                    thread_id,
                    err,
                    statuses,
                });
            }
        }
    }

    Ok(ChildStatusSubscriptions {
        status_rxs,
        terminal_statuses,
    })
}

async fn final_statuses_for_children(
    session: &crate::session::session::Session,
    receiver_agents: &[CollabAgentRef],
) -> HashMap<ThreadId, AgentStatus> {
    let mut statuses = HashMap::new();
    for receiver_agent in receiver_agents {
        let thread_id = receiver_agent.thread_id;
        let status = session.services.agent_control.get_status(thread_id).await;
        if is_final(&status) {
            statuses.insert(thread_id, status);
        }
    }
    statuses
}

enum WaitOutcome {
    AllTerminal,
    TimedOut,
}

/// Block until every enumerated child is terminal or the deadline elapses.
///
/// The wait wakes on a child status transition or a parent mailbox change, but only
/// *returns* when an authoritative re-poll shows all children terminal (or the deadline
/// fires). A mailbox change alone never ends the wait — that is what previously made
/// wait_agent spin at fan-out > 1, since a child-completion notification leaves pending
/// mail that fires the mailbox watch on every call.
///
/// Edge cases (intentional): if a `Completed` child is handed a new task and goes
/// `Running` again during the wait, the re-poll observes it as non-final and keeps
/// waiting (bounded by the deadline). A child stuck non-final after its status sender
/// drops pays the full timeout — its waiter is gone and `get_status` never turns final.
async fn wait_for_all_children_terminal(
    session: Arc<crate::session::session::Session>,
    status_rxs: Vec<(ThreadId, Receiver<AgentStatus>)>,
    terminal_statuses: &HashMap<ThreadId, AgentStatus>,
    mailbox_rx: &mut tokio::sync::watch::Receiver<()>,
    receiver_agents: &[CollabAgentRef],
    deadline: Instant,
) -> WaitOutcome {
    let mut status_waiters = FuturesUnordered::new();
    for (thread_id, status_rx) in status_rxs {
        // Skip children already terminal at entry — a waiter for them would resolve
        // instantly and just trigger a redundant re-poll. The authoritative
        // final_statuses_for_children poll below still counts them toward "all terminal".
        if terminal_statuses.contains_key(&thread_id) {
            continue;
        }
        status_waiters.push(wait_for_final_status(session.clone(), thread_id, status_rx));
    }

    let mut mailbox_open = true;
    loop {
        // Re-poll authoritatively on entry and after every wake. Return as soon as every
        // enumerated child is terminal — never on a mailbox change alone.
        let terminal = final_statuses_for_children(&session, receiver_agents).await;
        if receiver_agents
            .iter()
            .all(|agent| terminal.contains_key(&agent.thread_id))
        {
            return WaitOutcome::AllTerminal;
        }

        tokio::select! {
            // A child status transitioned -> loop and re-poll. The yielded value is
            // intentionally ignored; the re-poll above is the source of truth.
            _ = status_waiters.next(), if !status_waiters.is_empty() => continue,
            // Mailbox changed (e.g. a child-completion notification). Re-poll; do NOT
            // return early. Disable the arm if the sender dropped to avoid a busy spin.
            mailbox_changed = mailbox_rx.changed(), if mailbox_open => {
                if mailbox_changed.is_err() {
                    mailbox_open = false;
                }
                continue;
            }
            // Deadline arm is the only always-enabled branch -> select! can never have
            // all branches disabled.
            _ = sleep_until(deadline) => return WaitOutcome::TimedOut,
        }
    }
}

async fn wait_for_final_status(
    session: Arc<crate::session::session::Session>,
    thread_id: ThreadId,
    mut status_rx: Receiver<AgentStatus>,
) -> Option<(ThreadId, AgentStatus)> {
    let mut status = status_rx.borrow().clone();
    if is_final(&status) {
        return Some((thread_id, status));
    }

    loop {
        if status_rx.changed().await.is_err() {
            let latest = session.services.agent_control.get_status(thread_id).await;
            return is_final(&latest).then_some((thread_id, latest));
        }
        status = status_rx.borrow().clone();
        if is_final(&status) {
            return Some((thread_id, status));
        }
    }
}

async fn wait_for_mailbox_change(
    mailbox_rx: &mut tokio::sync::watch::Receiver<()>,
    deadline: Instant,
) -> bool {
    match timeout_at(deadline, mailbox_rx.changed()).await {
        Ok(Ok(())) => true,
        Ok(Err(_)) | Err(_) => false,
    }
}
