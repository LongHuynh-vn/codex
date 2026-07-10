use super::*;
use codex_extension_api::ExtensionData;
use codex_extension_api::TurnItemContributor;
use codex_protocol::items::AgentMessageContent;
use codex_protocol::protocol::Event;
use pretty_assertions::assert_eq;
use std::sync::Arc;

struct RewriteAgentMessageContributor;

#[async_trait::async_trait]
impl TurnItemContributor for RewriteAgentMessageContributor {
    async fn contribute(
        &self,
        _thread_store: &ExtensionData,
        _turn_store: &ExtensionData,
        item: &mut TurnItem,
    ) -> Result<(), String> {
        if let TurnItem::AgentMessage(agent_message) = item {
            agent_message.content = vec![AgentMessageContent::Text {
                text: "plan contributed assistant text".to_string(),
            }];
        }
        Ok(())
    }
}

fn assistant_output_text(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: Some("msg-1".to_string()),
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText {
            text: text.to_string(),
        }],
        phase: None,
    }
}

fn agent_message_item(id: &str, text: &str) -> codex_protocol::items::AgentMessageItem {
    codex_protocol::items::AgentMessageItem {
        id: id.to_string(),
        content: vec![AgentMessageContent::Text {
            text: text.to_string(),
        }],
        phase: None,
        memory_citation: None,
    }
}

fn pending_agent_message_item(id: &str) -> TurnItem {
    TurnItem::AgentMessage(agent_message_item(id, ""))
}

fn take_event_messages(rx: &async_channel::Receiver<Event>) -> Vec<EventMsg> {
    std::iter::from_fn(|| rx.try_recv().ok())
        .map(|event| event.msg)
        .collect()
}

#[test]
fn effective_turn_last_agent_message_uses_fallback_for_gemini_spawned_subagent() {
    assert_eq!(
        effective_turn_last_agent_message(
            WireApi::GeminiNative,
            /*is_spawned_subagent*/ true,
            /*terminal_last_agent_message*/ None,
            Some("report".to_string()),
            /*send_message_to_root_fallback*/ None,
        ),
        Some("report".to_string())
    );
}

#[test]
fn complete_task_result_renderer_scopes_status_prefix() {
    let cases = [
        (
            CompleteTaskStatus::Completed,
            "Sub-agent task completed:\nfinished",
        ),
        (
            CompleteTaskStatus::Partial,
            "Sub-agent task ended (partial):\nfinished",
        ),
        (
            CompleteTaskStatus::Failed,
            "Sub-agent task failed:\nfinished",
        ),
    ];

    for (status, expected) in cases {
        assert_eq!(
            render_complete_task_result(CompleteTaskResult {
                status,
                result: "finished".to_string(),
            }),
            expected
        );
    }
}

#[test]
fn complete_task_grace_turn_is_sampled_grounding_free() {
    // O27: a Gemini-native spawned sub-agent's complete_task grace turn (`suppress = true`) must be
    // sampled grounding-free regardless of the turn's configured search mode, so the child finalizes
    // from already gathered work instead of continuing research.
    for mode in [
        GeminiSearchMode::Grounding,
        GeminiSearchMode::Hybrid,
        GeminiSearchMode::Tavily,
        GeminiSearchMode::Off,
    ] {
        assert_eq!(
            prompt_gemini_search_mode(mode, /*suppress_gemini_grounding*/ true),
            Some(GeminiSearchMode::Off),
            "suppressed grace turn must force grounding-free for {mode:?}"
        );
    }
    // A non-grace turn (`suppress = false`) keeps its configured mode unchanged.
    for mode in [
        GeminiSearchMode::Grounding,
        GeminiSearchMode::Hybrid,
        GeminiSearchMode::Tavily,
        GeminiSearchMode::Off,
    ] {
        assert_eq!(
            prompt_gemini_search_mode(mode, /*suppress_gemini_grounding*/ false),
            Some(mode),
            "non-grace turn must keep its configured mode {mode:?}"
        );
    }
}

#[test]
fn effective_turn_last_agent_message_keeps_gemini_root_terminal_value() {
    assert_eq!(
        effective_turn_last_agent_message(
            WireApi::GeminiNative,
            /*is_spawned_subagent*/ false,
            /*terminal_last_agent_message*/ None,
            Some("report".to_string()),
            Some("send message report".to_string()),
        ),
        None
    );
}

#[test]
fn effective_turn_last_agent_message_keeps_non_gemini_terminal_value() {
    assert_eq!(
        effective_turn_last_agent_message(
            WireApi::Responses,
            /*is_spawned_subagent*/ true,
            /*terminal_last_agent_message*/ None,
            Some("report".to_string()),
            Some("send message report".to_string()),
        ),
        None
    );
}

#[test]
fn effective_turn_last_agent_message_prefers_terminal_value() {
    assert_eq!(
        effective_turn_last_agent_message(
            WireApi::GeminiNative,
            /*is_spawned_subagent*/ true,
            Some("final".to_string()),
            Some("report".to_string()),
            Some("send message report".to_string()),
        ),
        Some("final".to_string())
    );
}

#[test]
fn effective_turn_last_agent_message_uses_send_message_fallback_last() {
    assert_eq!(
        effective_turn_last_agent_message(
            WireApi::GeminiNative,
            /*is_spawned_subagent*/ true,
            /*terminal_last_agent_message*/ None,
            /*fallback_last_agent_message*/ None,
            Some("send message report".to_string()),
        ),
        Some("send message report".to_string())
    );
}

#[test]
fn effective_turn_last_agent_message_prefers_assistant_fallback_over_send_message() {
    assert_eq!(
        effective_turn_last_agent_message(
            WireApi::GeminiNative,
            /*is_spawned_subagent*/ true,
            /*terminal_last_agent_message*/ None,
            Some("assistant report".to_string()),
            Some("send message report".to_string()),
        ),
        Some("assistant report".to_string())
    );
}

#[tokio::test]
async fn plan_mode_uses_contributed_turn_item_for_last_agent_message() {
    let (mut session, turn_context) = crate::session::tests::make_session_and_context().await;
    let mut builder = codex_extension_api::ExtensionRegistryBuilder::new();
    builder.turn_item_contributor(Arc::new(RewriteAgentMessageContributor));
    session.services.extensions = Arc::new(builder.build());
    let turn_store = ExtensionData::new(turn_context.sub_id.clone());
    let mut state = PlanModeStreamState::new(&turn_context.sub_id, WireApi::Responses);
    let mut last_agent_message = None;
    let item = assistant_output_text("original assistant text");

    let handled = handle_assistant_item_done_in_plan_mode(
        &session,
        &turn_context,
        &turn_store,
        &item,
        &mut state,
        /*previously_active_item*/ None,
        &mut last_agent_message,
    )
    .await;

    assert!(handled);
    assert_eq!(
        last_agent_message.as_deref(),
        Some("plan contributed assistant text")
    );
}

#[tokio::test]
async fn gemini_plan_mode_suppresses_plan_bearing_agent_message_and_keeps_raw_item() {
    let (session, turn_context, rx) =
        crate::session::tests::make_session_and_context_with_rx().await;
    let turn_store = ExtensionData::new(turn_context.sub_id.clone());
    let item_id = "msg-1";
    let mut state = PlanModeStreamState::new(&turn_context.sub_id, WireApi::GeminiNative);
    state
        .pending_agent_message_items
        .insert(item_id.to_string(), pending_agent_message_item(item_id));

    handle_plan_segments(
        session.as_ref(),
        turn_context.as_ref(),
        &mut state,
        item_id,
        vec![
            ProposedPlanSegment::Normal("Intro\n".to_string()),
            ProposedPlanSegment::ProposedPlanStart,
            ProposedPlanSegment::ProposedPlanDelta("- step\n".to_string()),
            ProposedPlanSegment::ProposedPlanEnd,
            ProposedPlanSegment::Normal("Outro".to_string()),
        ],
    )
    .await;

    let item = assistant_output_text("Intro\n<proposed_plan>\n- step\n</proposed_plan>\nOutro");
    let mut last_agent_message = None;
    let handled = handle_assistant_item_done_in_plan_mode(
        session.as_ref(),
        turn_context.as_ref(),
        &turn_store,
        &item,
        &mut state,
        /*previously_active_item*/ None,
        &mut last_agent_message,
    )
    .await;

    assert!(handled);
    let events = take_event_messages(&rx);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, EventMsg::PlanDelta(_)))
    );
    assert!(events.iter().any(|event| matches!(
        event,
        EventMsg::ItemCompleted(item_completed) if matches!(&item_completed.item, TurnItem::Plan(_))
    )));
    assert!(!events.iter().any(|event| match event {
        EventMsg::AgentMessageContentDelta(_) => true,
        EventMsg::ItemStarted(item_started) => {
            matches!(item_started.item, TurnItem::AgentMessage(_))
        }
        EventMsg::ItemCompleted(item_completed) => {
            matches!(item_completed.item, TurnItem::AgentMessage(_))
        }
        _ => false,
    }));
    assert!(!state.pending_agent_message_items.contains_key(item_id));
    assert!(!state.started_agent_message_items.contains(item_id));
    assert!(!state.leading_whitespace_by_item.contains_key(item_id));
    assert_eq!(
        session.clone_history().await.raw_items().last(),
        Some(&item)
    );
}

#[tokio::test]
async fn gemini_plan_mode_keeps_grounding_message_before_plan() {
    let (session, turn_context, rx) =
        crate::session::tests::make_session_and_context_with_rx().await;
    let item_id = "grounding-msg";
    let mut state = PlanModeStreamState::new(&turn_context.sub_id, WireApi::GeminiNative);
    state
        .pending_agent_message_items
        .insert(item_id.to_string(), pending_agent_message_item(item_id));

    handle_plan_segments(
        session.as_ref(),
        turn_context.as_ref(),
        &mut state,
        item_id,
        vec![ProposedPlanSegment::Normal("Grounding answer".to_string())],
    )
    .await;
    assert!(take_event_messages(&rx).is_empty());

    emit_agent_message_in_plan_mode(
        session.as_ref(),
        turn_context.as_ref(),
        agent_message_item(item_id, "Grounding answer"),
        &mut state,
    )
    .await;

    let events = take_event_messages(&rx);
    assert!(events.iter().any(|event| matches!(
        event,
        EventMsg::ItemStarted(item_started) if matches!(&item_started.item, TurnItem::AgentMessage(item) if agent_message_text(item).is_empty())
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        EventMsg::ItemCompleted(item_completed) if matches!(&item_completed.item, TurnItem::AgentMessage(item) if agent_message_text(item) == "Grounding answer")
    )));
    assert!(!state.pending_agent_message_items.contains_key(item_id));
    assert!(!state.started_agent_message_items.contains(item_id));
}

#[tokio::test]
async fn gemini_plan_mode_cleans_already_started_agent_message_bookkeeping() {
    let (session, turn_context, rx) =
        crate::session::tests::make_session_and_context_with_rx().await;
    let item_id = "msg-1";
    let mut state = PlanModeStreamState::new(&turn_context.sub_id, WireApi::GeminiNative);
    state.plan_item_state.started = true;
    state
        .pending_agent_message_items
        .insert(item_id.to_string(), pending_agent_message_item(item_id));
    state
        .started_agent_message_items
        .insert(item_id.to_string());
    state
        .leading_whitespace_by_item
        .insert(item_id.to_string(), "\n".to_string());

    emit_agent_message_in_plan_mode(
        session.as_ref(),
        turn_context.as_ref(),
        agent_message_item(item_id, "suppressed prose"),
        &mut state,
    )
    .await;

    assert!(take_event_messages(&rx).is_empty());
    assert!(!state.pending_agent_message_items.contains_key(item_id));
    assert!(!state.started_agent_message_items.contains(item_id));
    assert!(!state.leading_whitespace_by_item.contains_key(item_id));
}

#[tokio::test]
async fn responses_plan_mode_keeps_normal_segments_and_agent_message() {
    let (session, turn_context, rx) =
        crate::session::tests::make_session_and_context_with_rx().await;
    let item_id = "msg-1";
    let mut state = PlanModeStreamState::new(&turn_context.sub_id, WireApi::Responses);
    state
        .pending_agent_message_items
        .insert(item_id.to_string(), pending_agent_message_item(item_id));

    handle_plan_segments(
        session.as_ref(),
        turn_context.as_ref(),
        &mut state,
        item_id,
        vec![
            ProposedPlanSegment::Normal("Intro".to_string()),
            ProposedPlanSegment::ProposedPlanStart,
            ProposedPlanSegment::ProposedPlanDelta("- step\n".to_string()),
            ProposedPlanSegment::ProposedPlanEnd,
            ProposedPlanSegment::Normal("Outro".to_string()),
        ],
    )
    .await;
    state
        .plan_item_state
        .complete_with_text(
            session.as_ref(),
            turn_context.as_ref(),
            "- step\n".to_string(),
        )
        .await;
    emit_agent_message_in_plan_mode(
        session.as_ref(),
        turn_context.as_ref(),
        agent_message_item(item_id, "IntroOutro"),
        &mut state,
    )
    .await;

    let events = take_event_messages(&rx);
    let agent_deltas = events
        .iter()
        .filter_map(|event| match event {
            EventMsg::AgentMessageContentDelta(delta) => Some(delta.delta.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(agent_deltas, "IntroOutro");
    assert!(events.iter().any(|event| matches!(
        event,
        EventMsg::ItemCompleted(item_completed) if matches!(&item_completed.item, TurnItem::AgentMessage(item) if agent_message_text(item) == "IntroOutro")
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        EventMsg::ItemCompleted(item_completed) if matches!(&item_completed.item, TurnItem::Plan(item) if item.text == "- step\n")
    )));
}

#[tokio::test]
async fn non_plan_mode_keeps_visible_text_deltas() {
    let (session, turn_context, rx) =
        crate::session::tests::make_session_and_context_with_rx().await;
    let mut parsers = AssistantMessageStreamParsers::new(/*plan_mode*/ false);
    let parsed = parsers.seed_item_text("msg-1", "visible text");

    emit_streamed_assistant_text_delta(
        session.as_ref(),
        turn_context.as_ref(),
        /*plan_mode_state*/ None,
        "msg-1",
        parsed,
    )
    .await;

    assert!(take_event_messages(&rx).iter().any(|event| matches!(
        event,
        EventMsg::AgentMessageContentDelta(delta) if delta.delta == "visible text"
    )));
}
