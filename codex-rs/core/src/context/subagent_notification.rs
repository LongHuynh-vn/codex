use codex_protocol::models::ContentItem;
use codex_protocol::models::MessagePhase;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::AgentStatus;

use super::ContextualUserFragment;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SubagentNotification {
    pub(crate) agent_reference: String,
    pub(crate) status: AgentStatus,
}

impl SubagentNotification {
    pub(crate) fn new(agent_reference: impl Into<String>, status: AgentStatus) -> Self {
        Self {
            agent_reference: agent_reference.into(),
            status,
        }
    }

    pub(crate) fn matches_clean_response_item(item: &ResponseItem) -> bool {
        let ResponseItem::Message {
            role,
            content,
            phase,
            ..
        } = item
        else {
            return false;
        };
        let [ContentItem::OutputText { text }] = content.as_slice() else {
            return false;
        };

        role == "assistant"
            && matches!(phase, Some(MessagePhase::Commentary))
            && Self::matches_text(text)
    }
}

impl ContextualUserFragment for SubagentNotification {
    fn role(&self) -> &'static str {
        "user"
    }

    fn markers(&self) -> (&'static str, &'static str) {
        Self::type_markers()
    }

    fn type_markers() -> (&'static str, &'static str) {
        ("<subagent_notification>", "</subagent_notification>")
    }

    fn body(&self) -> String {
        format!(
            "\n{}\n",
            serde_json::json!({
                "agent_path": &self.agent_reference,
                "status": &self.status,
            })
        )
    }
}
