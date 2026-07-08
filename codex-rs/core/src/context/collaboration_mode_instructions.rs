use super::ContextualUserFragment;
use codex_collaboration_mode_templates::PLAN;
use codex_collaboration_mode_templates::plan_for_gemini;
use codex_protocol::config_types::CollaborationMode;
use codex_protocol::config_types::ModeKind;
use codex_protocol::protocol::COLLABORATION_MODE_CLOSE_TAG;
use codex_protocol::protocol::COLLABORATION_MODE_OPEN_TAG;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CollaborationModeInstructions {
    instructions: String,
}

impl CollaborationModeInstructions {
    pub(crate) fn from_collaboration_mode(
        collaboration_mode: &CollaborationMode,
        is_gemini: bool,
    ) -> Option<Self> {
        let instructions = collaboration_mode
            .settings
            .developer_instructions
            .as_ref()
            .filter(|instructions| !instructions.is_empty())?;
        let instructions = if is_gemini
            && collaboration_mode.mode == ModeKind::Plan
            && instructions.as_str() == PLAN
        {
            plan_for_gemini().to_string()
        } else {
            instructions.clone()
        };

        Some(Self { instructions })
    }
}

impl ContextualUserFragment for CollaborationModeInstructions {
    fn role(&self) -> &'static str {
        "developer"
    }

    fn markers(&self) -> (&'static str, &'static str) {
        Self::type_markers()
    }

    fn type_markers() -> (&'static str, &'static str) {
        (COLLABORATION_MODE_OPEN_TAG, COLLABORATION_MODE_CLOSE_TAG)
    }

    fn body(&self) -> String {
        self.instructions.clone()
    }
}

#[cfg(test)]
#[path = "collaboration_mode_instructions_tests.rs"]
mod tests;
