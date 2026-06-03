use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::openai_models::ConfigShellToolType;
use codex_protocol::openai_models::InputModality;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ModelVisibility;
use codex_protocol::openai_models::ModelsResponse;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::openai_models::ReasoningEffortPreset;
use codex_protocol::openai_models::TruncationPolicyConfig;
use codex_protocol::protocol::MultiAgentVersion;

pub const GEMINI_3_5_FLASH_MODEL: &str = "gemini-3.5-flash";
pub const GEMINI_3_1_PRO_PREVIEW_MODEL: &str = "gemini-3.1-pro-preview";

pub fn gemini_model_catalog() -> ModelsResponse {
    ModelsResponse {
        models: vec![
            gemini_model(GEMINI_3_5_FLASH_MODEL, "Gemini 3.5 Flash", 1_000_000, 1),
            gemini_model(
                GEMINI_3_1_PRO_PREVIEW_MODEL,
                "Gemini 3.1 Pro Preview",
                1_000_000,
                2,
            ),
        ],
    }
}

fn gemini_model(slug: &str, display_name: &str, context_window: i64, priority: i32) -> ModelInfo {
    ModelInfo {
        slug: slug.to_string(),
        display_name: display_name.to_string(),
        description: None,
        default_reasoning_level: Some(ReasoningEffort::High),
        supported_reasoning_levels: vec![
            reasoning_preset(ReasoningEffort::Minimal),
            reasoning_preset(ReasoningEffort::Low),
            reasoning_preset(ReasoningEffort::Medium),
            reasoning_preset(ReasoningEffort::High),
        ],
        shell_type: ConfigShellToolType::Default,
        visibility: ModelVisibility::List,
        supported_in_api: true,
        priority,
        additional_speed_tiers: Vec::new(),
        service_tiers: Vec::new(),
        default_service_tier: None,
        availability_nux: None,
        upgrade: None,
        base_instructions: String::new(),
        model_messages: None,
        supports_reasoning_summaries: true,
        default_reasoning_summary: ReasoningSummary::Auto,
        support_verbosity: false,
        default_verbosity: None,
        apply_patch_tool_type: None,
        web_search_tool_type: Default::default(),
        truncation_policy: TruncationPolicyConfig::tokens(context_window),
        supports_parallel_tool_calls: false,
        supports_image_detail_original: false,
        context_window: Some(context_window),
        max_context_window: Some(context_window),
        auto_compact_token_limit: Some(900_000),
        effective_context_window_percent: 90,
        experimental_supported_tools: Vec::new(),
        input_modalities: vec![InputModality::Text, InputModality::Image],
        used_fallback_model_metadata: false,
        supports_search_tool: false,
        auto_review_model_override: None,
        tool_mode: None,
        multi_agent_version: Some(MultiAgentVersion::V1),
    }
}

fn reasoning_preset(effort: ReasoningEffort) -> ReasoningEffortPreset {
    ReasoningEffortPreset {
        effort,
        description: String::new(),
    }
}
