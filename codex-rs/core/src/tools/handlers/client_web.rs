use std::collections::BTreeMap;
use std::collections::HashSet;

use codex_protocol::items::TurnItem;
use codex_protocol::items::WebSearchItem;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::WebSearchAction;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolName;
use codex_tools::ToolSpec;
use codex_utils_string::take_bytes_at_char_boundary;
use http::StatusCode;
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
const MAX_SEARCH_LIMIT: usize = 20;
const DEFAULT_TAVILY_SEARCH_DEPTH: &str = "advanced";
const DEFAULT_TAVILY_CHUNKS_PER_SOURCE: usize = 3;
const TAVILY_INCLUDE_RAW_CONTENT: &str = "markdown";
const PER_RESULT_BODY_MAX_BYTES: usize = 2_000;
const MAX_IMAGES_PER_RESULT: usize = 3;
const MAX_INCLUDE_DOMAINS: usize = 300;
const MAX_EXCLUDE_DOMAINS: usize = 150;
const TAVILY_SEARCH_DEPTH_ENV: &str = "CODEX_TAVILY_SEARCH_DEPTH";
const TAVILY_DEFAULT_SEARCH_LIMIT_ENV: &str = "CODEX_TAVILY_DEFAULT_SEARCH_LIMIT";
const DEFAULT_FETCH_MAX_BYTES: usize = 20_000;
const MAX_FETCH_MAX_BYTES: usize = 50_000;
const MAX_IMAGES_ON_PAGE: usize = 20;

#[derive(Clone, Debug)]
pub(crate) struct ClientWebConfig {
    tavily_api_key: Option<String>,
    tavily_search_url: String,
    tavily_search_depth: String,
    default_search_limit: usize,
}

impl ClientWebConfig {
    pub(crate) fn from_env() -> Self {
        Self {
            tavily_api_key: std::env::var(TAVILY_API_KEY_ENV)
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty()),
            tavily_search_url: TAVILY_SEARCH_URL.to_string(),
            tavily_search_depth: parse_tavily_search_depth(
                std::env::var(TAVILY_SEARCH_DEPTH_ENV).ok().as_deref(),
            ),
            default_search_limit: parse_tavily_default_search_limit(
                std::env::var(TAVILY_DEFAULT_SEARCH_LIMIT_ENV)
                    .ok()
                    .as_deref(),
            ),
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
    #[serde(default)]
    time_range: Option<String>,
    #[serde(default)]
    start_date: Option<String>,
    #[serde(default)]
    end_date: Option<String>,
    #[serde(default)]
    topic: Option<String>,
    #[serde(default)]
    include_domains: Option<Vec<String>>,
    #[serde(default)]
    exclude_domains: Option<Vec<String>>,
    #[serde(default)]
    exact_match: Option<bool>,
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
    search_depth: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    chunks_per_source: Option<usize>,
    include_raw_content: &'static str,
    include_images: bool,
    include_usage: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    time_range: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    start_date: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    end_date: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    topic: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include_domains: Option<&'a [String]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exclude_domains: Option<&'a [String]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exact_match: Option<bool>,
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
    score: Option<f64>,
    raw_content: Option<String>,
    published_date: Option<String>,
    #[serde(default)]
    images: Vec<TavilyImage>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TavilyImage {
    Url(String),
    Obj { url: String },
}

impl TavilyImage {
    fn url(&self) -> &str {
        match self {
            TavilyImage::Url(url) | TavilyImage::Obj { url } => url,
        }
    }
}

#[async_trait::async_trait]
impl ToolExecutor<ToolInvocation> for ClientWebSearchHandler {
    fn tool_name(&self) -> ToolName {
        ToolName::plain(WEB_SEARCH_TOOL_NAME)
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec::Function(ResponsesApiTool {
            name: WEB_SEARCH_TOOL_NAME.to_string(),
            description: concat!(
                "Search the web through Codex's client-side Tavily backend. ",
                "For latest, newest, or current questions, set time_range to \"month\" or ",
                "\"year\" to avoid stale results. Prefer include_domains for primary or ",
                "official sources. Use exact_match for precise version strings, model ",
                "names, benchmark names, or quoted phrases."
            )
            .to_string(),
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
                            "Maximum number of results to return, up to 20.".to_string(),
                        )),
                    ),
                    (
                        "time_range".to_string(),
                        JsonSchema::string_enum(
                            vec![
                                serde_json::json!("day"),
                                serde_json::json!("week"),
                                serde_json::json!("month"),
                                serde_json::json!("year"),
                            ],
                            Some(
                                "Optional recency filter. Use month or year for latest/current questions."
                                    .to_string(),
                            ),
                        ),
                    ),
                    (
                        "start_date".to_string(),
                        JsonSchema::string(Some(
                            "Optional start date filter in YYYY-MM-DD format.".to_string(),
                        )),
                    ),
                    (
                        "end_date".to_string(),
                        JsonSchema::string(Some(
                            "Optional end date filter in YYYY-MM-DD format.".to_string(),
                        )),
                    ),
                    (
                        "topic".to_string(),
                        JsonSchema::string_enum(
                            vec![
                                serde_json::json!("general"),
                                serde_json::json!("news"),
                                serde_json::json!("finance"),
                            ],
                            Some(
                                "Optional search topic. Use news for current events when publication dates matter."
                                    .to_string(),
                            ),
                        ),
                    ),
                    (
                        "include_domains".to_string(),
                        JsonSchema::array(
                            JsonSchema::string(/*description*/ None),
                            Some(
                                "Optional domains to restrict results to; prefer official sources."
                                    .to_string(),
                            ),
                        ),
                    ),
                    (
                        "exclude_domains".to_string(),
                        JsonSchema::array(
                            JsonSchema::string(/*description*/ None),
                            Some("Optional domains to exclude from results.".to_string()),
                        ),
                    ),
                    (
                        "exact_match".to_string(),
                        JsonSchema::boolean(Some(
                            "Require exact quoted phrase matches for precise names or versions."
                                .to_string(),
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
            .unwrap_or(self.config.default_search_limit)
            .min(MAX_SEARCH_LIMIT);
        if limit == 0 {
            return Err(FunctionCallError::RespondToModel(
                "limit must be greater than zero".to_string(),
            ));
        }
        validate_domain_count(
            "include_domains",
            args.include_domains.as_deref(),
            MAX_INCLUDE_DOMAINS,
        )?;
        validate_domain_count(
            "exclude_domains",
            args.exclude_domains.as_deref(),
            MAX_EXCLUDE_DOMAINS,
        )?;

        let item = TurnItem::WebSearch(WebSearchItem {
            id: call_id,
            query: query.clone(),
            action: WebSearchAction::Search {
                query: Some(query.clone()),
                queries: None,
            },
        });
        session.emit_turn_item_started(turn.as_ref(), &item).await;

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
                search_depth: self.config.tavily_search_depth.as_str(),
                chunks_per_source: (self.config.tavily_search_depth == DEFAULT_TAVILY_SEARCH_DEPTH)
                    .then_some(DEFAULT_TAVILY_CHUNKS_PER_SOURCE),
                include_raw_content: TAVILY_INCLUDE_RAW_CONTENT,
                include_images: true,
                include_usage: true,
                time_range: args.time_range.as_deref(),
                start_date: args.start_date.as_deref(),
                end_date: args.end_date.as_deref(),
                topic: args.topic.as_deref(),
                include_domains: args.include_domains.as_deref(),
                exclude_domains: args.exclude_domains.as_deref(),
                exact_match: args.exact_match,
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
                return Err(FunctionCallError::RespondToModel(format_tavily_error(
                    status, &url, &body,
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

        session.emit_turn_item_completed(turn.as_ref(), item).await;
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
        let item = TurnItem::WebSearch(WebSearchItem {
            id: call_id,
            query: url_text.clone(),
            action: WebSearchAction::OpenPage {
                url: Some(url_text),
            },
        });
        session.emit_turn_item_started(turn.as_ref(), &item).await;

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
                let image_urls = extract_image_urls(&body, &url);
                append_image_urls(truncate_text(&html_to_text(&body), max_bytes), image_urls)
            } else {
                truncate_text(&body, max_bytes)
            };
            let output =
                format!("URL: {url}\nStatus: {status}\nContent-Type: {content_type}\n\n{body}");
            Ok(boxed_tool_output(FunctionToolOutput::from_content(
                vec![FunctionCallOutputContentItem::InputText { text: output }],
                Some(true),
            )))
        }
        .await;

        session.emit_turn_item_completed(turn.as_ref(), item).await;
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
        lines.push(format!("{}. {title}", index + 1));
        if !url.is_empty() {
            lines.push(format!("   URL: {url}"));
        }
        if let Some(score) = result.score {
            lines.push(format!("   Score: {score:.3}"));
        }
        if let Some(published_date) = result.published_date.as_deref()
            && !published_date.trim().is_empty()
        {
            lines.push(format!("   Published: {}", published_date.trim()));
        }
        if let Some(body) = result_body(result) {
            lines.push("   Body:".to_string());
            lines.extend(body.lines().map(|line| format!("   {line}")));
        }
        let image_urls = result
            .images
            .iter()
            .map(TavilyImage::url)
            .map(str::trim)
            .filter(|url| !url.is_empty())
            .take(MAX_IMAGES_PER_RESULT)
            .collect::<Vec<_>>();
        if !image_urls.is_empty() {
            lines.push("   Images:".to_string());
            lines.extend(image_urls.into_iter().map(|url| format!("   - {url}")));
        }
    }
    truncate_text(&lines.join("\n"), DEFAULT_FETCH_MAX_BYTES)
}

fn result_body(result: &TavilySearchResult) -> Option<String> {
    if let Some(raw_content) = result.raw_content.as_deref()
        && !raw_content.trim().is_empty()
    {
        return Some(truncate_text(raw_content.trim(), PER_RESULT_BODY_MAX_BYTES));
    }

    let content = result.content.as_deref()?;
    let content = normalize_whitespace(content);
    if content.is_empty() {
        None
    } else {
        Some(truncate_text(&content, PER_RESULT_BODY_MAX_BYTES))
    }
}

fn format_tavily_error(status: StatusCode, url: &Url, body: &str) -> String {
    let detail = match status.as_u16() {
        400 => "invalid Tavily request",
        401 => "Tavily authentication failed; check TAVILY_API_KEY",
        429 => "Tavily rate limit exceeded",
        432 => "Tavily plan or API key quota exceeded",
        433 => "Tavily pay-as-you-go limit exceeded",
        _ if status.is_server_error() => "Tavily server error",
        _ => "Tavily request failed",
    };
    let body = truncate_text(body, DEFAULT_FETCH_MAX_BYTES);
    format!("web_search {detail} for {url} with status {status}: {body}")
}

fn validate_domain_count(
    name: &str,
    domains: Option<&[String]>,
    max: usize,
) -> Result<(), FunctionCallError> {
    if let Some(domains) = domains
        && domains.len() > max
    {
        return Err(FunctionCallError::RespondToModel(format!(
            "{name} supports at most {max} domains"
        )));
    }
    Ok(())
}

fn parse_tavily_search_depth(raw: Option<&str>) -> String {
    let Some(value) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return DEFAULT_TAVILY_SEARCH_DEPTH.to_string();
    };
    let value = value.to_ascii_lowercase();
    match value.as_str() {
        "advanced" | "basic" | "fast" | "ultra-fast" => value,
        _ => DEFAULT_TAVILY_SEARCH_DEPTH.to_string(),
    }
}

fn parse_tavily_default_search_limit(raw: Option<&str>) -> usize {
    raw.map(str::trim)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .map(|value| value.min(MAX_SEARCH_LIMIT))
        .unwrap_or(DEFAULT_SEARCH_LIMIT)
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

fn append_image_urls(mut body: String, image_urls: Vec<String>) -> String {
    if image_urls.is_empty() {
        return body;
    }
    if !body.is_empty() {
        body.push_str("\n\n");
    }
    body.push_str("Images on page:");
    for image_url in image_urls {
        body.push_str("\n- ");
        body.push_str(&image_url);
    }
    body
}

fn extract_image_urls(input: &str, base_url: &Url) -> Vec<String> {
    let mut urls = Vec::new();
    let mut seen = HashSet::new();
    let lowercase = input.to_ascii_lowercase();
    let mut offset = 0;

    while let Some(relative_start) = lowercase[offset..].find("<img") {
        let start = offset + relative_start;
        let after_name = start + "<img".len();
        if !is_img_tag_boundary(lowercase.as_bytes(), after_name) {
            offset = after_name;
            continue;
        }
        let relative_end = lowercase[start..]
            .find('>')
            .unwrap_or(lowercase.len() - start);
        let end = start + relative_end;
        let tag = &input[start..end];
        if let Some(src) = extract_src_attr(tag)
            && let Ok(url) = base_url.join(src.trim())
            && matches!(url.scheme(), "http" | "https")
        {
            let url = url.to_string();
            if seen.insert(url.clone()) {
                urls.push(url);
                if urls.len() >= MAX_IMAGES_ON_PAGE {
                    break;
                }
            }
        }
        offset = (end + 1).min(lowercase.len());
    }

    urls
}

fn is_img_tag_boundary(bytes: &[u8], index: usize) -> bool {
    match bytes.get(index) {
        None => true,
        Some(b'/' | b'>') => true,
        Some(byte) => byte.is_ascii_whitespace(),
    }
}

fn extract_src_attr(tag: &str) -> Option<String> {
    let bytes = tag.as_bytes();
    let mut index = "<img".len();
    while index < bytes.len() {
        while index < bytes.len()
            && (bytes[index].is_ascii_whitespace() || matches!(bytes[index], b'/' | b'>'))
        {
            index += 1;
        }
        let name_start = index;
        while index < bytes.len()
            && !bytes[index].is_ascii_whitespace()
            && !matches!(bytes[index], b'=' | b'/' | b'>')
        {
            index += 1;
        }
        if name_start == index {
            break;
        }
        let name = &tag[name_start..index];
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if bytes.get(index) != Some(&b'=') {
            continue;
        }
        index += 1;
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        let value = if matches!(bytes.get(index), Some(b'"' | b'\'')) {
            let quote = bytes[index];
            index += 1;
            let value_start = index;
            while index < bytes.len() && bytes[index] != quote {
                index += 1;
            }
            let value = &tag[value_start..index];
            if index < bytes.len() {
                index += 1;
            }
            value
        } else {
            let value_start = index;
            while index < bytes.len() && !bytes[index].is_ascii_whitespace() && bytes[index] != b'>'
            {
                index += 1;
            }
            &tag[value_start..index]
        };
        if name.eq_ignore_ascii_case("src") {
            return Some(decode_basic_html_entities(value.trim()));
        }
    }
    None
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
