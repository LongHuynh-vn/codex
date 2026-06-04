use std::collections::BTreeMap;

use codex_protocol::models::FunctionCallOutputContentItem;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolName;
use codex_tools::ToolSpec;
use codex_utils_string::take_bytes_at_char_boundary;
use http::header::CONTENT_TYPE;
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use url::Url;

use crate::function_tool::FunctionCallError;
use crate::tools::context::FunctionToolOutput;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::context::boxed_tool_output;
use crate::tools::handlers::parse_arguments;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::ToolExecutor;

const WEB_SEARCH_TOOL_NAME: &str = "web_search";
const WEB_FETCH_TOOL_NAME: &str = "web_fetch";
const WEB_SEARCH_BACKEND_ENV: &str = "CODEX_GEMINI_WEB_SEARCH_URL";
const DEFAULT_SEARCH_LIMIT: usize = 5;
const MAX_SEARCH_LIMIT: usize = 10;
const DEFAULT_FETCH_MAX_BYTES: usize = 20_000;
const MAX_FETCH_MAX_BYTES: usize = 50_000;

#[derive(Clone, Debug, Default)]
pub(crate) struct ClientWebConfig {
    pub(crate) search_endpoint: Option<String>,
}

impl ClientWebConfig {
    pub(crate) fn from_env() -> Self {
        Self {
            search_endpoint: std::env::var(WEB_SEARCH_BACKEND_ENV)
                .ok()
                .filter(|value| !value.trim().is_empty()),
        }
    }
}

pub(crate) struct ClientWebSearchHandler {
    client: Client,
    config: ClientWebConfig,
}

impl ClientWebSearchHandler {
    pub(crate) fn new(client: Client, config: ClientWebConfig) -> Self {
        Self { client, config }
    }
}

pub(crate) struct ClientWebFetchHandler {
    client: Client,
}

impl ClientWebFetchHandler {
    pub(crate) fn new(client: Client) -> Self {
        Self { client }
    }
}

#[derive(Debug, Deserialize)]
struct WebSearchArgs {
    query: String,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct WebFetchArgs {
    url: String,
    #[serde(default)]
    max_bytes: Option<usize>,
}

#[async_trait::async_trait]
impl ToolExecutor<ToolInvocation> for ClientWebSearchHandler {
    fn tool_name(&self) -> ToolName {
        ToolName::plain(WEB_SEARCH_TOOL_NAME)
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec::Function(ResponsesApiTool {
            name: WEB_SEARCH_TOOL_NAME.to_string(),
            description: "Search the web through Codex's client-side HTTP backend.".to_string(),
            strict: false,
            defer_loading: None,
            parameters: JsonSchema::object(
                BTreeMap::from([
                    (
                        "query".to_string(),
                        JsonSchema::string(Some("Search query.".to_string())),
                    ),
                    (
                        "limit".to_string(),
                        JsonSchema::integer(Some(
                            "Maximum number of results to return, up to 10.".to_string(),
                        )),
                    ),
                ]),
                Some(vec!["query".to_string()]),
                None,
            ),
            output_schema: None,
        })
    }

    async fn handle(
        &self,
        invocation: ToolInvocation,
    ) -> Result<Box<dyn crate::tools::context::ToolOutput>, FunctionCallError> {
        let ToolPayload::Function { arguments } = invocation.payload else {
            return Err(FunctionCallError::RespondToModel(
                "web_search handler received unsupported payload".to_string(),
            ));
        };
        let args: WebSearchArgs = parse_arguments(&arguments)?;
        let query = args.query.trim();
        if query.is_empty() {
            return Err(FunctionCallError::RespondToModel(
                "query must not be empty".to_string(),
            ));
        }
        let Some(endpoint) = self.config.search_endpoint.as_deref() else {
            return Err(FunctionCallError::RespondToModel(format!(
                "web_search backend is not configured; set {WEB_SEARCH_BACKEND_ENV}"
            )));
        };

        let limit = args
            .limit
            .unwrap_or(DEFAULT_SEARCH_LIMIT)
            .min(MAX_SEARCH_LIMIT);
        if limit == 0 {
            return Err(FunctionCallError::RespondToModel(
                "limit must be greater than zero".to_string(),
            ));
        }
        let mut url = Url::parse(endpoint).map_err(|err| {
            FunctionCallError::RespondToModel(format!(
                "configured web_search endpoint is not a valid URL: {err}"
            ))
        })?;
        url.query_pairs_mut()
            .append_pair("q", query)
            .append_pair("format", "json");

        let response = self.client.get(url.clone()).send().await.map_err(|err| {
            FunctionCallError::RespondToModel(format!("web_search request failed: {err}"))
        })?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let body = truncate_text(&body, DEFAULT_FETCH_MAX_BYTES);
            return Err(FunctionCallError::RespondToModel(format!(
                "web_search request to {url} failed with {status}: {body}"
            )));
        }

        let body: Value = response.json().await.map_err(|err| {
            FunctionCallError::RespondToModel(format!("web_search response was not JSON: {err}"))
        })?;
        let output = format_search_results(&body, limit);
        Ok(boxed_tool_output(FunctionToolOutput::from_text(
            output,
            Some(true),
        )))
    }
}

impl CoreToolRuntime for ClientWebSearchHandler {}

#[async_trait::async_trait]
impl ToolExecutor<ToolInvocation> for ClientWebFetchHandler {
    fn tool_name(&self) -> ToolName {
        ToolName::plain(WEB_FETCH_TOOL_NAME)
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec::Function(ResponsesApiTool {
            name: WEB_FETCH_TOOL_NAME.to_string(),
            description: "Fetch a web page or document through Codex's client-side HTTP backend."
                .to_string(),
            strict: false,
            defer_loading: None,
            parameters: JsonSchema::object(
                BTreeMap::from([
                    (
                        "url".to_string(),
                        JsonSchema::string(Some("HTTP or HTTPS URL to fetch.".to_string())),
                    ),
                    (
                        "max_bytes".to_string(),
                        JsonSchema::integer(Some(
                            "Maximum response bytes to return, up to 50000.".to_string(),
                        )),
                    ),
                ]),
                Some(vec!["url".to_string()]),
                None,
            ),
            output_schema: None,
        })
    }

    async fn handle(
        &self,
        invocation: ToolInvocation,
    ) -> Result<Box<dyn crate::tools::context::ToolOutput>, FunctionCallError> {
        let ToolPayload::Function { arguments } = invocation.payload else {
            return Err(FunctionCallError::RespondToModel(
                "web_fetch handler received unsupported payload".to_string(),
            ));
        };
        let args: WebFetchArgs = parse_arguments(&arguments)?;
        let url = Url::parse(args.url.trim()).map_err(|err| {
            FunctionCallError::RespondToModel(format!("url must be a valid URL: {err}"))
        })?;
        match url.scheme() {
            "http" | "https" => {}
            scheme => {
                return Err(FunctionCallError::RespondToModel(format!(
                    "unsupported URL scheme `{scheme}`; use http or https"
                )));
            }
        }
        let max_bytes = args
            .max_bytes
            .unwrap_or(DEFAULT_FETCH_MAX_BYTES)
            .min(MAX_FETCH_MAX_BYTES);
        if max_bytes == 0 {
            return Err(FunctionCallError::RespondToModel(
                "max_bytes must be greater than zero".to_string(),
            ));
        }

        let response = self.client.get(url.clone()).send().await.map_err(|err| {
            FunctionCallError::RespondToModel(format!("web_fetch request failed: {err}"))
        })?;
        let status = response.status();
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("unknown")
            .to_string();
        let body = response.text().await.map_err(|err| {
            FunctionCallError::RespondToModel(format!("web_fetch response read failed: {err}"))
        })?;
        let body = if content_type.to_ascii_lowercase().contains("html") {
            html_to_text(&body)
        } else {
            body
        };
        let body = truncate_text(&body, max_bytes);
        let output =
            format!("URL: {url}\nStatus: {status}\nContent-Type: {content_type}\n\n{body}");
        Ok(boxed_tool_output(FunctionToolOutput::from_content(
            vec![FunctionCallOutputContentItem::InputText { text: output }],
            Some(true),
        )))
    }
}

impl CoreToolRuntime for ClientWebFetchHandler {}

fn format_search_results(body: &Value, limit: usize) -> String {
    let Some(results) = body.get("results").and_then(Value::as_array) else {
        return "No results.".to_string();
    };
    if results.is_empty() {
        return "No results.".to_string();
    }

    let mut lines = vec![format!(
        "Search results ({} shown):",
        results.len().min(limit)
    )];
    for (index, result) in results.iter().take(limit).enumerate() {
        let title = string_field(result, &["title", "name"]).unwrap_or("Untitled");
        let url = string_field(result, &["url", "href", "link"]).unwrap_or("");
        let snippet =
            string_field(result, &["content", "snippet", "description"]).unwrap_or_default();
        lines.push(format!("{}. {title}", index + 1));
        if !url.is_empty() {
            lines.push(format!("   URL: {url}"));
        }
        if !snippet.is_empty() {
            lines.push(format!("   Snippet: {}", normalize_whitespace(snippet)));
        }
    }
    truncate_text(&lines.join("\n"), DEFAULT_FETCH_MAX_BYTES)
}

fn string_field<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(Value::as_str))
}

fn html_to_text(input: &str) -> String {
    let mut output = String::new();
    let mut in_tag = false;
    for ch in input.chars() {
        match ch {
            '<' => {
                in_tag = true;
                output.push(' ');
            }
            '>' => in_tag = false,
            _ if !in_tag => output.push(ch),
            _ => {}
        }
    }
    normalize_whitespace(&decode_basic_html_entities(&output))
}

fn decode_basic_html_entities(input: &str) -> String {
    input
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

fn normalize_whitespace(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_text(text: &str, max_bytes: usize) -> String {
    let truncated = take_bytes_at_char_boundary(text, max_bytes);
    if truncated.len() == text.len() {
        truncated.to_string()
    } else {
        format!("{truncated}\n[truncated]")
    }
}

#[cfg(test)]
#[path = "client_web_tests.rs"]
mod tests;
