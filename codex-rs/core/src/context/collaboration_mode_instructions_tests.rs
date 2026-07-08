use super::CollaborationModeInstructions;
use crate::context::ContextualUserFragment;
use codex_collaboration_mode_templates::PLAN;
use codex_collaboration_mode_templates::PLAN_GEMINI_ADDENDUM;
use codex_collaboration_mode_templates::plan_for_gemini;
use codex_protocol::config_types::CollaborationMode;
use codex_protocol::config_types::ModeKind;
use codex_protocol::config_types::Settings;
use pretty_assertions::assert_eq;

const PLAN_QUALITY_CONTRACT_MARKER: &str = "## Plan quality contract (strict)";

fn collaboration_mode(mode: ModeKind, instructions: Option<&str>) -> CollaborationMode {
    CollaborationMode {
        mode,
        settings: Settings {
            model: "gpt-5.4".to_string(),
            reasoning_effort: None,
            developer_instructions: instructions.map(str::to_string),
        },
    }
}

fn rendered_body(collaboration_mode: &CollaborationMode, is_gemini: bool) -> Option<String> {
    CollaborationModeInstructions::from_collaboration_mode(collaboration_mode, is_gemini)
        .map(|instructions| instructions.body())
}

#[test]
fn non_gemini_plan_mode_renders_stock_plan_bytes() {
    let collaboration_mode = collaboration_mode(ModeKind::Plan, Some(PLAN));

    assert_eq!(
        rendered_body(&collaboration_mode, false),
        Some(PLAN.to_string())
    );
}

#[test]
fn gemini_plan_mode_stock_text_gets_quality_contract() {
    let collaboration_mode = collaboration_mode(ModeKind::Plan, Some(PLAN));

    let body = rendered_body(&collaboration_mode, true).expect("body should render");
    assert_eq!(body, plan_for_gemini());
    assert!(body.starts_with(PLAN));
    assert_eq!(body.matches(PLAN_QUALITY_CONTRACT_MARKER).count(), 1);
}

#[test]
fn gemini_plan_mode_custom_instructions_untouched() {
    let custom_instructions = "Draft plans with source citations.";
    let collaboration_mode = collaboration_mode(ModeKind::Plan, Some(custom_instructions));

    assert_eq!(
        rendered_body(&collaboration_mode, true),
        Some(custom_instructions.to_string())
    );
}

#[test]
fn gemini_default_mode_stock_text_untouched() {
    let collaboration_mode = collaboration_mode(ModeKind::Default, Some(PLAN));

    assert_eq!(
        rendered_body(&collaboration_mode, true),
        Some(PLAN.to_string())
    );
}

#[test]
fn gemini_plan_mode_empty_instructions_render_none() {
    let empty = collaboration_mode(ModeKind::Plan, Some(""));
    let absent = collaboration_mode(ModeKind::Plan, None);

    assert_eq!(rendered_body(&empty, true), None);
    assert_eq!(rendered_body(&absent, true), None);
}

#[test]
fn plan_for_gemini_is_stock_plan_plus_addendum() {
    let expected_suffix = format!("\n{PLAN_GEMINI_ADDENDUM}");

    assert_eq!(
        plan_for_gemini().strip_prefix(PLAN),
        Some(expected_suffix.as_str())
    );
}
