use crate::config::DEFAULT_MULTI_AGENT_V2_SUBAGENT_USAGE_HINT_TEXT;
use crate::session::turn_context::TurnContext;
use codex_model_provider_info::WireApi;
use codex_protocol::protocol::MultiAgentVersion;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::SubAgentSource;

pub(crate) const GEMINI_MULTI_AGENT_V2_SUBAGENT_USAGE_HINT_TEXT: &str = r#"You are a worker agent completing one delegated task.

You cannot spawn sub-agents or wait on other agents. Complete the assigned task yourself. Coordinate with other agents only through `send_message`.

When you finish, call `complete_task` with a self-contained final report and status `completed`, `partial`, or `failed`. Do not end on a trailing assistant message.
"#;

pub(crate) fn subagent_usage_hint_text_for_wire_api(
    wire_api: WireApi,
    subagent_usage_hint_text: &str,
) -> &str {
    if wire_api == WireApi::GeminiNative
        && subagent_usage_hint_text == DEFAULT_MULTI_AGENT_V2_SUBAGENT_USAGE_HINT_TEXT
    {
        GEMINI_MULTI_AGENT_V2_SUBAGENT_USAGE_HINT_TEXT
    } else {
        subagent_usage_hint_text
    }
}

pub(super) fn usage_hint_text<'a>(
    turn_context: &'a TurnContext,
    session_source: &SessionSource,
) -> Option<&'a str> {
    if turn_context.multi_agent_version != MultiAgentVersion::V2 {
        return None;
    }

    let multi_agent_v2 = &turn_context.config.multi_agent_v2;
    if !multi_agent_v2.usage_hint_enabled {
        return None;
    }

    match session_source {
        SessionSource::SubAgent(SubAgentSource::ThreadSpawn { .. }) => multi_agent_v2
            .subagent_usage_hint_text
            .as_deref()
            .map(|hint| {
                subagent_usage_hint_text_for_wire_api(turn_context.provider.info().wire_api, hint)
            }),
        SessionSource::Cli
        | SessionSource::VSCode
        | SessionSource::Exec
        | SessionSource::Mcp
        | SessionSource::Custom(_)
        | SessionSource::Unknown => multi_agent_v2.root_agent_usage_hint_text.as_deref(),
        SessionSource::Internal(_) | SessionSource::SubAgent(_) => None,
    }
}
