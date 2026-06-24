use super::*;
use codex_extension_api::ExtensionData;
use codex_extension_api::TurnItemContributor;
use codex_protocol::items::AgentMessageContent;
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
fn empty_report_retry_turn_is_sampled_grounding_free() {
    // O27: a Gemini-native spawned sub-agent's empty-report retry turn (`suppress = true`) must be
    // sampled grounding-free regardless of the turn's configured search mode, so the retry does not
    // re-hit the grounding-correlated empty `STOP`.
    for mode in [
        GeminiSearchMode::Grounding,
        GeminiSearchMode::Hybrid,
        GeminiSearchMode::Tavily,
        GeminiSearchMode::Off,
    ] {
        assert_eq!(
            prompt_gemini_search_mode(mode, /*suppress_gemini_grounding*/ true),
            Some(GeminiSearchMode::Off),
            "suppressed retry must force grounding-free for {mode:?}"
        );
    }
    // A non-retry turn (`suppress = false`) keeps its configured mode unchanged.
    for mode in [
        GeminiSearchMode::Grounding,
        GeminiSearchMode::Hybrid,
        GeminiSearchMode::Tavily,
        GeminiSearchMode::Off,
    ] {
        assert_eq!(
            prompt_gemini_search_mode(mode, /*suppress_gemini_grounding*/ false),
            Some(mode),
            "non-retry turn must keep its configured mode {mode:?}"
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
    let mut state = PlanModeStreamState::new(&turn_context.sub_id);
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
