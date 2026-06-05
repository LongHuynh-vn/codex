use std::collections::BTreeMap;

use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::WebSearchAction;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::WebSearchBeginEvent;
use codex_protocol::protocol::WebSearchEndEvent;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolName;
use codex_tools::ToolSpec;
use codex_utils_string::take_bytes_at_char_boundary;
use http::header::CONTENT_TYPE;
use reqwest::Client;
use serde::Deserialize;
use serde::Serialize;
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
const TAVILY_API_KEY_ENV: &str = "TAVILY_API_KEY";
const TAVILY_SEARCH_URL: &str = "https://api.tavily.com/search";
const DEFAULT_SEARCH_LIMIT: usize = 5;
const MAX_SEARCH_LIMIT: usize = 10;
const DEFAULT_FETCH_MAX_BYTES: usize = 20_000;
const MAX_FETCH_MAX_BYTES: usize = 50_000;

#[derive(Clone, Debug)]
pub(crate) struct ClientWebConfig {
    tavily_api_key: Option<String>,
    tavily_search_url: String,
}

impl ClientWebConfig {
    pub(crate) fn from_env() -> Self {
        Self {
            tavily_api_key: std::env::var(TAVILY_API_KEY_ENV)
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty()),
            tavily_search_url: TAVILY_SEARCH_URL.to_string(),
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

#[derive(Debug, Serialize)]
struct TavilySearchRequest<'a> {
    query: &'a str,
    max_results: usize,
    search_depth: &'static str,
    topic: &'static str,
}

#[derive(Debug, Deserialize)]
struct TavilySearchResponse {
    #[serde(default)]
    results: Vec<TavilySearchResult>,
}

#[derive(Debug, Deserialize)]
struct TavilySearchResult {
    title: Option<String>,
    url: Option<String>,
    content: Option<String>,
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
        let ToolInvocation {
            session,
            turn,
            call_id,
            payload,
            ..
        } = invocation;
        let ToolPayload::Function { arguments } = payload else {
            return Err(FunctionCallError::RespondToModel(
                "web_search handler received unsupported payload".to_string(),
            ));
        };
        let args: WebSearchArgs = parse_arguments(&arguments)?;
        let query = args.query.trim().to_string();
        if query.is_empty() {
            return Err(FunctionCallError::RespondToModel(
                "query must not be empty".to_string(),
            ));
        }

        let limit = args
            .limit
            .unwrap_or(DEFAULT_SEARCH_LIMIT)
            .min(MAX_SEARCH_LIMIT);
        if limit == 0 {
            return Err(FunctionCallError::RespondToModel(
                "limit must be greater than zero".to_string(),
            ));
        }

        let action = WebSearchAction::Search {
            query: Some(query.clone()),
            queries: None,
        };
        session
            .send_event(
                turn.as_ref(),
                EventMsg::WebSearchBegin(WebSearchBeginEvent {
                    call_id: call_id.clone(),
                }),
            )
            .await;

        let result = async {
            let Some(api_key) = self.config.tavily_api_key.as_deref() else {
                return Err(FunctionCallError::RespondToModel(format!(
                    "web_search is not configured; set {TAVILY_API_KEY_ENV} to use Tavily search"
                )));
            };
            let url = Url::parse(&self.config.tavily_search_url).map_err(|err| {
                FunctionCallError::RespondToModel(format!(
                    "configured Tavily search URL is not valid: {err}"
                ))
            })?;
            let request = TavilySearchRequest {
                query: &query,
                max_results: limit,
                search_depth: "basic",
                topic: "general",
            };

            let response = self
                .client
                .post(url.clone())
                .bearer_auth(api_key)
                .json(&request)
                .send()
                .await
                .map_err(|err| {
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

            let body: TavilySearchResponse = response.json().await.map_err(|err| {
                FunctionCallError::RespondToModel(format!(
                    "web_search response was not JSON: {err}"
                ))
            })?;
            let output = format_search_results(&body, limit);
            Ok(boxed_tool_output(FunctionToolOutput::from_text(
                output,
                Some(true),
            )))
        }
        .await;

        session
            .send_event(
                turn.as_ref(),
                EventMsg::WebSearchEnd(WebSearchEndEvent {
                    call_id,
                    query,
                    action,
                }),
            )
            .await;
        result
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
        let ToolInvocation {
            session,
            turn,
            call_id,
            payload,
            ..
        } = invocation;
        let ToolPayload::Function { arguments } = payload else {
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

        let url_text = url.to_string();
        let action = WebSearchAction::OpenPage {
            url: Some(url_text.clone()),
        };
        session
            .send_event(
                turn.as_ref(),
                EventMsg::WebSearchBegin(WebSearchBeginEvent {
                    call_id: call_id.clone(),
                }),
            )
            .await;

        let result = async {
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
        .await;

        session
            .send_event(
                turn.as_ref(),
                EventMsg::WebSearchEnd(WebSearchEndEvent {
                    call_id,
                    query: url_text,
                    action,
                }),
            )
            .await;
        result
    }
}

impl CoreToolRuntime for ClientWebFetchHandler {}

fn format_search_results(body: &TavilySearchResponse, limit: usize) -> String {
    if body.results.is_empty() {
        return "No results.".to_string();
    }

    let mut lines = vec![format!(
        "Search results ({} shown):",
        body.results.len().min(limit)
    )];
    for (index, result) in body.results.iter().take(limit).enumerate() {
        let title = result.title.as_deref().unwrap_or("Untitled");
        let url = result.url.as_deref().unwrap_or("");
        let snippet = result.content.as_deref().unwrap_or_default();
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
