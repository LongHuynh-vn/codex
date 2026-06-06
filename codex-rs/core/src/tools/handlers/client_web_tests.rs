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
            "search_depth": "basic",
            "topic": "general"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                {
                    "title": "Paris forecast",
                    "url": "https://example.com/paris",
                    "content": "Clear skies"
                },
                {
                    "title": "Paris climate",
                    "url": "https://example.com/climate",
                    "content": "Temperate"
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
                    "Search results (2 shown):\n1. Paris forecast\n   URL: https://example.com/paris\n   Snippet: Clear skies\n2. Paris climate\n   URL: https://example.com/climate\n   Snippet: Temperate"
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
            tavily_search_url: format!("{}/search", server.uri()),
        },
    );
    let (invocation, rx) =
        invocation_with_rx(WEB_SEARCH_TOOL_NAME, json!({"query": "weather paris"})).await;
    let result = handler.handle(invocation).await;

    let Err(FunctionCallError::RespondToModel(message)) = result else {
        panic!("expected Tavily error");
    };
    assert!(
        message.contains("failed with 500 Internal Server Error"),
        "unexpected error message: {message}"
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
