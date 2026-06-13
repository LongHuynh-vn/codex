pub(crate) use codex_utils_absolute_path::test_support::PathBufExt;
pub(crate) use codex_utils_absolute_path::test_support::test_path_buf;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::legacy_core::config::Config;
use codex_model_provider_info::OPENAI_PROVIDER_ID;

/// Pin the model provider deterministically for tests so results don't depend on ambient
/// `GEMINI_API_KEY` / `GOOGLE_GENAI_USE_VERTEXAI` in the developer's shell. Defaults to the
/// OpenAI built-in provider, which the TUI test suite is written against. A test that needs a
/// different provider sets `config.model_provider`/`config.model_provider_id` explicitly after
/// constructing the config (as the bedrock/openai-proxy status snapshots already do).
pub(crate) fn pin_default_test_provider(config: &mut Config) {
    config.model_provider_id = OPENAI_PROVIDER_ID.to_string();
    config.model_provider = config
        .model_providers
        .get(OPENAI_PROVIDER_ID)
        .cloned()
        .expect("openai built-in provider present in test config");
}

pub(crate) fn test_path_display(path: &str) -> String {
    test_path_buf(path).display().to_string()
}

pub(crate) fn session_source_cli<T>() -> T
where
    T: DeserializeOwned,
{
    from_app_server_wire(codex_app_server_protocol::SessionSource::Cli)
}

pub(crate) fn skill_scope_user<T>() -> T
where
    T: DeserializeOwned,
{
    from_app_server_wire(codex_app_server_protocol::SkillScope::User)
}

pub(crate) fn skill_scope_repo<T>() -> T
where
    T: DeserializeOwned,
{
    from_app_server_wire(codex_app_server_protocol::SkillScope::Repo)
}

fn from_app_server_wire<T>(value: impl Serialize) -> T
where
    T: DeserializeOwned,
{
    serde_json::to_value(value)
        .and_then(serde_json::from_value)
        .unwrap_or_else(|err| {
            panic!("app-server wire value should map to legacy helper type: {err}")
        })
}
