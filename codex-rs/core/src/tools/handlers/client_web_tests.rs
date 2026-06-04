use std::sync::Arc;

use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseInputItem;
use pretty_assertions::assert_eq;
use reqwest::Client;
use serde_json::json;
use tokio::sync::Mutex;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;
use wiremock::matchers::query_param;

use super::*;
use crate::session::tests::make_session_and_context;
use crate::tools::context::ToolCallSource;
use crate::turn_diff_tracker::TurnDiffTracker;

async fn invocation(tool_name: &str, arguments: serde_json::Value) -> ToolInvocation {
    let (session, turn) = make_session_and_context().await;
    ToolInvocation {
        session: session.into(),
        turn: turn.into(),
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
    Mock::given(method("GET"))
        .and(path("/"))
        .and(query_param("q", "weather paris"))
        .and(query_param("format", "json"))
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
                    "snippet": "Temperate"
                }
            ]
        })))
        .mount(&server)
        .await;

    let handler = ClientWebSearchHandler::new(
        Client::new(),
        ClientWebConfig {
            search_endpoint: Some(server.uri()),
        },
    );
    let payload = ToolPayload::Function {
        arguments: json!({"query": "weather paris", "limit": 2}).to_string(),
    };
    let output = handler
        .handle(
            invocation(
                WEB_SEARCH_TOOL_NAME,
                json!({"query": "weather paris", "limit": 2}),
            )
            .await,
        )
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
        arguments: json!({"url": url, "max_bytes": 2000}).to_string(),
    };
    let output = handler
        .handle(invocation(WEB_FETCH_TOOL_NAME, json!({"url": url, "max_bytes": 2000})).await)
        .await
        .expect("web fetch should succeed");
    let text = function_output_text(output.to_response_item("call-web-fetch", &payload));

    assert!(
        text.contains(
            "Status: 200 OK\nContent-Type: text/html; charset=utf-8\n\nTitle Hello & welcome."
        ),
        "unexpected web_fetch output: {text}"
    );
}
