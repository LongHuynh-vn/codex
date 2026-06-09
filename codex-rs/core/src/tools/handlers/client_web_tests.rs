use std::sync::Arc;

use codex_protocol::items::TurnItem;
use codex_protocol::items::WebSearchItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::WebSearchAction;
use codex_protocol::protocol::Event;
use codex_protocol::protocol::EventMsg;
use pretty_assertions::assert_eq;
use reqwest::Client;
use serde_json::json;
use tokio::sync::Mutex;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::body_json;
use wiremock::matchers::header;
use wiremock::matchers::method;
use wiremock::matchers::path;

use super::*;
use crate::session::tests::make_session_and_context_with_rx;
use crate::tools::context::ToolCallSource;
use crate::turn_diff_tracker::TurnDiffTracker;

async fn invocation_with_rx(
    tool_name: &str,
    arguments: serde_json::Value,
) -> (ToolInvocation, async_channel::Receiver<Event>) {
    let (session, turn, rx) = make_session_and_context_with_rx().await;
    (tool_invocation(tool_name, arguments, session, turn), rx)
}

fn tool_invocation(
    tool_name: &str,
    arguments: serde_json::Value,
    session: std::sync::Arc<crate::session::session::Session>,
    turn: std::sync::Arc<crate::session::turn_context::TurnContext>,
) -> ToolInvocation {
    ToolInvocation {
        session,
        turn,
        cancellation_token: tokio_util::sync::CancellationToken::new(),
        tracker: Arc::new(Mutex::new(TurnDiffTracker::new())),
        call_id: format!("call-{tool_name}"),
        tool_name: ToolName::plain(tool_name),
        source: ToolCallSource::Direct,
        payload: ToolPayload::Function {
            arguments: serde_json::to_string(&arguments).expect("arguments should serialize"),
        },
    }
}

async fn assert_web_search_item_events(
    rx: &async_channel::Receiver<Event>,
    expected_call_id: &str,
    expected_query: &str,
    expected_action: WebSearchAction,
) {
    let mut messages = Vec::new();
    loop {
        let event = rx.recv().await.expect("web search item event");
        let is_completed = matches!(
            &event.msg,
            EventMsg::ItemCompleted(completed)
                if matches!(&completed.item, TurnItem::WebSearch(_))
        );
        messages.push(event.msg);
        if is_completed {
            break;
        }
    }
    while let Ok(event) = rx.try_recv() {
        messages.push(event.msg);
    }

    let expected_item = WebSearchItem {
        id: expected_call_id.to_string(),
        query: expected_query.to_string(),
        action: expected_action,
    };
    let started_items = messages
        .iter()
        .filter_map(|message| match message {
            EventMsg::ItemStarted(started) => match &started.item {
                TurnItem::WebSearch(item) => Some(item),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    let completed_items = messages
        .iter()
        .filter_map(|message| match message {
            EventMsg::ItemCompleted(completed) => match &completed.item {
                TurnItem::WebSearch(item) => Some(item),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(started_items, vec![&expected_item]);
    assert_eq!(completed_items, vec![&expected_item]);
}

fn function_output_text(item: ResponseInputItem) -> String {
    let ResponseInputItem::FunctionCallOutput { output, .. } = item else {
        panic!("expected function call output");
    };
    output
        .text_content()
        .expect("function output should be text")
        .to_string()
}

#[tokio::test]
async fn web_search_executes_client_side_backend() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/search"))
        .and(header("authorization", "Bearer test-key"))
        .and(header("content-type", "application/json"))
        .and(body_json(json!({
            "query": "weather paris",
            "max_results": 2,
            "search_depth": "advanced",
            "chunks_per_source": 3,
            "include_raw_content": "markdown",
            "include_images": true,
            "include_usage": true
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                {
                    "title": "Paris forecast",
                    "url": "https://example.com/paris",
                    "content": "Fallback content should not be shown",
                    "raw_content": "Clear skies\nWind: light",
                    "score": 0.98765,
                    "published_date": "2026-06-09",
                    "images": [
                        "https://example.com/paris.jpg",
                        {
                            "url": "https://example.com/paris-map.jpg",
                            "description": "Map"
                        }
                    ]
                },
                {
                    "title": "Paris climate",
                    "url": "https://example.com/climate",
                    "content": "Temperate climate",
                    "score": 0.5
                }
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let handler = ClientWebSearchHandler::new(
        Client::new(),
        ClientWebConfig {
            tavily_api_key: Some("test-key".to_string()),
            tavily_search_url: format!("{}/search", server.uri()),
            tavily_search_depth: DEFAULT_TAVILY_SEARCH_DEPTH.to_string(),
            default_search_limit: DEFAULT_SEARCH_LIMIT,
        },
    );
    let payload = ToolPayload::Function {
        arguments: json!({"query": "weather paris", "limit": 2}).to_string(),
    };
    let (invocation, rx) = invocation_with_rx(
        WEB_SEARCH_TOOL_NAME,
        json!({"query": "weather paris", "limit": 2}),
    )
    .await;
    let output = handler
        .handle(invocation)
        .await
        .expect("web search should succeed");
    let item = output.to_response_item("call-web-search", &payload);

    assert_eq!(
        item,
        ResponseInputItem::FunctionCallOutput {
            call_id: "call-web-search".to_string(),
            output: FunctionCallOutputPayload {
                body: codex_protocol::models::FunctionCallOutputBody::Text(
                    "Search results (2 shown):\n1. Paris forecast\n   URL: https://example.com/paris\n   Score: 0.988\n   Published: 2026-06-09\n   Body:\n   Clear skies\n   Wind: light\n   Images:\n   - https://example.com/paris.jpg\n   - https://example.com/paris-map.jpg\n2. Paris climate\n   URL: https://example.com/climate\n   Score: 0.500\n   Body:\n   Temperate climate"
                        .to_string()
                ),
                success: Some(true),
            },
        }
    );
    assert_web_search_item_events(
        &rx,
        "call-web_search",
        "weather paris",
        WebSearchAction::Search {
            query: Some("weather paris".to_string()),
            queries: None,
        },
    )
    .await;
}

#[tokio::test]
async fn web_search_requires_tavily_api_key() {
    let handler = ClientWebSearchHandler::new(
        Client::new(),
        ClientWebConfig {
            tavily_api_key: None,
            tavily_search_url: "http://127.0.0.1/search".to_string(),
            tavily_search_depth: DEFAULT_TAVILY_SEARCH_DEPTH.to_string(),
            default_search_limit: DEFAULT_SEARCH_LIMIT,
        },
    );
    let (invocation, rx) =
        invocation_with_rx(WEB_SEARCH_TOOL_NAME, json!({"query": "weather paris"})).await;
    let result = handler.handle(invocation).await;

    let Err(FunctionCallError::RespondToModel(message)) = result else {
        panic!("expected missing Tavily key error");
    };
    assert_eq!(
        message,
        "web_search is not configured; set TAVILY_API_KEY to use Tavily search"
    );
    assert_web_search_item_events(
        &rx,
        "call-web_search",
        "weather paris",
        WebSearchAction::Search {
            query: Some("weather paris".to_string()),
            queries: None,
        },
    )
    .await;
}

#[tokio::test]
async fn web_search_emits_item_completed_when_tavily_returns_error() {
    let server = MockServer::start().await;
    let url = format!("{}/search", server.uri());
    Mock::given(method("POST"))
        .and(path("/search"))
        .respond_with(ResponseTemplate::new(500).set_body_string("upstream unavailable"))
        .expect(1)
        .mount(&server)
        .await;

    let handler = ClientWebSearchHandler::new(
        Client::new(),
        ClientWebConfig {
            tavily_api_key: Some("test-key".to_string()),
            tavily_search_url: url.clone(),
            tavily_search_depth: DEFAULT_TAVILY_SEARCH_DEPTH.to_string(),
            default_search_limit: DEFAULT_SEARCH_LIMIT,
        },
    );
    let (invocation, rx) =
        invocation_with_rx(WEB_SEARCH_TOOL_NAME, json!({"query": "weather paris"})).await;
    let result = handler.handle(invocation).await;

    let Err(FunctionCallError::RespondToModel(message)) = result else {
        panic!("expected Tavily error");
    };
    assert_eq!(
        message,
        format!(
            "web_search Tavily server error for {url} with status 500 Internal Server Error: upstream unavailable"
        )
    );
    assert_web_search_item_events(
        &rx,
        "call-web_search",
        "weather paris",
        WebSearchAction::Search {
            query: Some("weather paris".to_string()),
            queries: None,
        },
    )
    .await;
}

#[tokio::test]
async fn web_search_serializes_recency_and_source_filters() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/search"))
        .and(body_json(json!({
            "query": "OpenAI GPT-5 benchmark",
            "max_results": 1,
            "search_depth": "advanced",
            "chunks_per_source": 3,
            "include_raw_content": "markdown",
            "include_images": true,
            "include_usage": true,
            "time_range": "month",
            "start_date": "2026-01-01",
            "end_date": "2026-02-01",
            "topic": "news",
            "include_domains": ["openai.com"],
            "exclude_domains": ["example.com"],
            "exact_match": true
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": []
        })))
        .expect(1)
        .mount(&server)
        .await;

    let handler = ClientWebSearchHandler::new(
        Client::new(),
        ClientWebConfig {
            tavily_api_key: Some("test-key".to_string()),
            tavily_search_url: format!("{}/search", server.uri()),
            tavily_search_depth: DEFAULT_TAVILY_SEARCH_DEPTH.to_string(),
            default_search_limit: DEFAULT_SEARCH_LIMIT,
        },
    );
    let payload = ToolPayload::Function {
        arguments: json!({
            "query": "OpenAI GPT-5 benchmark",
            "limit": 1,
            "time_range": "month",
            "start_date": "2026-01-01",
            "end_date": "2026-02-01",
            "topic": "news",
            "include_domains": ["openai.com"],
            "exclude_domains": ["example.com"],
            "exact_match": true
        })
        .to_string(),
    };
    let (invocation, _) = invocation_with_rx(
        WEB_SEARCH_TOOL_NAME,
        json!({
            "query": "OpenAI GPT-5 benchmark",
            "limit": 1,
            "time_range": "month",
            "start_date": "2026-01-01",
            "end_date": "2026-02-01",
            "topic": "news",
            "include_domains": ["openai.com"],
            "exclude_domains": ["example.com"],
            "exact_match": true
        }),
    )
    .await;
    let output = handler
        .handle(invocation)
        .await
        .expect("web search should succeed");
    let text = function_output_text(output.to_response_item("call-web-search", &payload));

    assert_eq!(text, "No results.");
}

#[tokio::test]
async fn web_search_omits_chunks_per_source_for_non_advanced_depth() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/search"))
        .and(body_json(json!({
            "query": "weather paris",
            "max_results": 5,
            "search_depth": "fast",
            "include_raw_content": "markdown",
            "include_images": true,
            "include_usage": true
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": []
        })))
        .expect(1)
        .mount(&server)
        .await;

    let handler = ClientWebSearchHandler::new(
        Client::new(),
        ClientWebConfig {
            tavily_api_key: Some("test-key".to_string()),
            tavily_search_url: format!("{}/search", server.uri()),
            tavily_search_depth: "fast".to_string(),
            default_search_limit: DEFAULT_SEARCH_LIMIT,
        },
    );
    let (invocation, _) =
        invocation_with_rx(WEB_SEARCH_TOOL_NAME, json!({"query": "weather paris"})).await;
    let result = handler.handle(invocation).await;

    assert!(result.is_ok(), "web search should succeed");
}

#[test]
fn web_search_deserializes_string_and_object_images() {
    let response: TavilySearchResponse = serde_json::from_value(json!({
        "results": [
            {
                "title": "Image result",
                "url": "https://example.com",
                "content": "Result body",
                "images": [
                    "https://example.com/one.png",
                    {
                        "url": "https://example.com/two.png",
                        "description": "Second"
                    }
                ]
            }
        ]
    }))
    .expect("Tavily response should deserialize mixed image shapes");

    let output = format_search_results(&response, 1);

    assert!(
        output.contains("   - https://example.com/one.png"),
        "missing string image URL: {output}"
    );
    assert!(
        output.contains("   - https://example.com/two.png"),
        "missing object image URL: {output}"
    );
}

#[test]
fn web_search_formats_raw_content_and_caps_body_and_images() {
    let raw_content = format!("raw {}", "x".repeat(PER_RESULT_BODY_MAX_BYTES + 20));
    let response = TavilySearchResponse {
        results: vec![TavilySearchResult {
            title: Some("Rich result".to_string()),
            url: Some("https://example.com/rich".to_string()),
            content: Some("fallback content should not be shown".to_string()),
            score: Some(0.1234),
            raw_content: Some(raw_content),
            published_date: Some("2026-06-09".to_string()),
            images: vec![
                TavilyImage::Url("https://example.com/1.png".to_string()),
                TavilyImage::Obj {
                    url: "https://example.com/2.png".to_string(),
                },
                TavilyImage::Url("https://example.com/3.png".to_string()),
                TavilyImage::Url("https://example.com/4.png".to_string()),
            ],
        }],
    };

    let output = format_search_results(&response, 1);

    assert!(
        output.contains("   Body:\n   raw "),
        "missing raw body: {output}"
    );
    assert!(
        !output.contains("fallback content should not be shown"),
        "content fallback should not be shown when raw_content is present: {output}"
    );
    assert!(
        output.contains("[truncated]"),
        "body should be truncated: {output}"
    );
    assert!(
        output.contains("https://example.com/3.png"),
        "third image should be included: {output}"
    );
    assert!(
        !output.contains("https://example.com/4.png"),
        "fourth image should be capped: {output}"
    );
}

#[test]
fn tavily_error_messages_distinguish_statuses() {
    let url = Url::parse("https://api.tavily.com/search").expect("url should parse");
    let cases = [
        (400, "invalid Tavily request"),
        (401, "Tavily authentication failed; check TAVILY_API_KEY"),
        (429, "Tavily rate limit exceeded"),
        (432, "Tavily plan or API key quota exceeded"),
        (433, "Tavily pay-as-you-go limit exceeded"),
        (500, "Tavily server error"),
    ];

    for (status, expected) in cases {
        let status = StatusCode::from_u16(status).expect("status code should be valid");
        let message = format_tavily_error(status, &url, "body");
        assert!(
            message.contains(expected),
            "message should contain {expected:?}: {message}"
        );
        assert!(
            message.contains(&format!("status {status}")),
            "message should include status {status}: {message}"
        );
        assert!(
            message.ends_with(": body"),
            "message should include response body: {message}"
        );
    }
}

#[test]
fn tavily_search_depth_parser_validates_and_falls_back() {
    assert_eq!(parse_tavily_search_depth(None), "advanced");
    assert_eq!(parse_tavily_search_depth(Some("")), "advanced");
    assert_eq!(parse_tavily_search_depth(Some(" FAST ")), "fast");
    assert_eq!(parse_tavily_search_depth(Some("ultra-fast")), "ultra-fast");
    assert_eq!(parse_tavily_search_depth(Some("deep")), "advanced");
}

#[test]
fn tavily_default_search_limit_parser_validates_clamps_and_falls_back() {
    assert_eq!(
        parse_tavily_default_search_limit(None),
        DEFAULT_SEARCH_LIMIT
    );
    assert_eq!(
        parse_tavily_default_search_limit(Some("")),
        DEFAULT_SEARCH_LIMIT
    );
    assert_eq!(
        parse_tavily_default_search_limit(Some("0")),
        DEFAULT_SEARCH_LIMIT
    );
    assert_eq!(
        parse_tavily_default_search_limit(Some("invalid")),
        DEFAULT_SEARCH_LIMIT
    );
    assert_eq!(parse_tavily_default_search_limit(Some("12")), 12);
    assert_eq!(
        parse_tavily_default_search_limit(Some("100")),
        MAX_SEARCH_LIMIT
    );
}

#[tokio::test]
async fn web_fetch_executes_client_side_http_get() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/page"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            "<html><body><h1>Title</h1><p>Hello &amp; welcome.</p></body></html>",
            "text/html; charset=utf-8",
        ))
        .mount(&server)
        .await;

    let url = format!("{}/page", server.uri());
    let handler = ClientWebFetchHandler::new(Client::new());
    let payload = ToolPayload::Function {
        arguments: json!({"url": url.clone(), "max_bytes": 2000}).to_string(),
    };
    let (invocation, rx) = invocation_with_rx(
        WEB_FETCH_TOOL_NAME,
        json!({"url": url.clone(), "max_bytes": 2000}),
    )
    .await;
    let output = handler
        .handle(invocation)
        .await
        .expect("web fetch should succeed");
    let text = function_output_text(output.to_response_item("call-web-fetch", &payload));

    assert!(
        text.contains(
            "Status: 200 OK\nContent-Type: text/html; charset=utf-8\n\nTitle Hello & welcome."
        ),
        "unexpected web_fetch output: {text}"
    );
    assert!(
        !text.contains("Images on page:"),
        "unexpected image list: {text}"
    );
    assert_web_search_item_events(
        &rx,
        "call-web_fetch",
        &url,
        WebSearchAction::OpenPage {
            url: Some(url.clone()),
        },
    )
    .await;
}

#[tokio::test]
async fn web_fetch_appends_image_urls_from_html() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/reports/page.html"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            r#"
            <html>
              <body>
                <h1>Report</h1>
                <img src="chart.png">
                <IMG SRC='/figures/plot.webp'>
                <img src="https://cdn.example.com/abs.jpg">
                <img src='//assets.example.com/protocol.png'>
                <img data-src="/lazy.png">
                <img src="chart.png">
                <img src=unquoted.gif>
              </body>
            </html>
            "#,
            "text/html; charset=utf-8",
        ))
        .mount(&server)
        .await;

    let url = format!("{}/reports/page.html", server.uri());
    let handler = ClientWebFetchHandler::new(Client::new());
    let payload = ToolPayload::Function {
        arguments: json!({"url": url.clone(), "max_bytes": 2000}).to_string(),
    };
    let (invocation, _) = invocation_with_rx(
        WEB_FETCH_TOOL_NAME,
        json!({"url": url.clone(), "max_bytes": 2000}),
    )
    .await;
    let output = handler
        .handle(invocation)
        .await
        .expect("web fetch should succeed");
    let text = function_output_text(output.to_response_item("call-web-fetch", &payload));

    let relative_chart = format!("{}/reports/chart.png", server.uri());
    assert!(
        text.contains("\n\nImages on page:\n- "),
        "missing image list: {text}"
    );
    assert!(
        text.contains(&format!("- {relative_chart}")),
        "missing relative image URL: {text}"
    );
    assert!(
        text.contains(&format!("- {}/figures/plot.webp", server.uri())),
        "missing root-relative image URL: {text}"
    );
    assert!(
        text.contains("- https://cdn.example.com/abs.jpg"),
        "missing absolute image URL: {text}"
    );
    assert!(
        text.contains("- http://assets.example.com/protocol.png"),
        "missing protocol-relative image URL: {text}"
    );
    assert!(
        text.contains(&format!("- {}/reports/unquoted.gif", server.uri())),
        "missing unquoted image URL: {text}"
    );
    assert_eq!(text.matches(&relative_chart).count(), 1);
    assert!(
        !text.contains("/lazy.png"),
        "data-src should be ignored: {text}"
    );
}

#[test]
fn web_fetch_image_url_extraction_caps_and_deduplicates() {
    let base = Url::parse("https://example.com/dir/page.html").expect("base URL should parse");
    let html = (0..25)
        .map(|index| format!(r#"<img src="image-{index}.png"><img src="image-{index}.png">"#))
        .collect::<String>();

    let image_urls = extract_image_urls(&html, &base);

    assert_eq!(image_urls.len(), MAX_IMAGES_ON_PAGE);
    assert_eq!(image_urls[0], "https://example.com/dir/image-0.png");
    assert_eq!(image_urls[19], "https://example.com/dir/image-19.png");
    assert!(!image_urls.iter().any(|url| url.ends_with("image-20.png")));
}

#[tokio::test]
async fn web_fetch_emits_item_completed_when_request_fails() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener should bind");
    let addr = listener
        .local_addr()
        .expect("test listener should have local address");
    let accept_task = tokio::spawn(async move {
        let (socket, _) = listener
            .accept()
            .await
            .expect("test listener should accept one connection");
        drop(socket);
    });

    let url = format!("http://{addr}/page");
    let handler = ClientWebFetchHandler::new(Client::new());
    let (invocation, rx) =
        invocation_with_rx(WEB_FETCH_TOOL_NAME, json!({"url": url.clone()})).await;
    let result = handler.handle(invocation).await;
    accept_task
        .await
        .expect("test listener task should complete");

    let Err(FunctionCallError::RespondToModel(message)) = result else {
        panic!("expected web_fetch request error");
    };
    assert!(
        message.starts_with("web_fetch request failed:"),
        "unexpected error message: {message}"
    );
    assert_web_search_item_events(
        &rx,
        "call-web_fetch",
        &url,
        WebSearchAction::OpenPage {
            url: Some(url.clone()),
        },
    )
    .await;
}
