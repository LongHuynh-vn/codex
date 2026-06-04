#![allow(clippy::expect_used)]

use std::ffi::OsString;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use anyhow::Result;
use codex_gemini_adapter::GEMINI_3_5_FLASH_MODEL;
use codex_gemini_adapter::model_config::gemini_model_catalog;
use codex_model_provider_info::GEMINI_PROVIDER_ID;
use codex_model_provider_info::ModelProviderInfo;
use codex_protocol::models::PermissionProfile;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodexBuilder;
use core_test_support::test_codex::TestCodexHarness;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use serial_test::serial;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::Respond;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;
use wiremock::matchers::path_regex;

const WEB_SEARCH_BACKEND_ENV: &str = "CODEX_GEMINI_WEB_SEARCH_URL";

struct EnvVarGuard {
    key: &'static str,
    original: Option<OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &std::ffi::OsStr) -> Self {
        let original = std::env::var_os(key);
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.original {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct GeminiRequestLog {
    requests: Arc<Mutex<Vec<Value>>>,
}

impl GeminiRequestLog {
    pub(super) fn requests(&self) -> Vec<Value> {
        self.requests.lock().expect("request log lock").clone()
    }
}

struct GeminiSseResponder {
    num_calls: AtomicUsize,
    responses: Vec<String>,
    requests: Arc<Mutex<Vec<Value>>>,
}

impl Respond for GeminiSseResponder {
    fn respond(&self, request: &wiremock::Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).expect("Gemini request JSON");
        self.requests.lock().expect("request log lock").push(body);

        let call_num = self.num_calls.fetch_add(1, Ordering::SeqCst);
        let body = self
            .responses
            .get(call_num)
            .unwrap_or_else(|| panic!("no Gemini response for call {call_num}"));
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(body.clone())
    }
}

pub(super) async fn mount_gemini_sse_sequence(
    server: &MockServer,
    responses: Vec<String>,
) -> GeminiRequestLog {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let responder = GeminiSseResponder {
        num_calls: AtomicUsize::new(0),
        responses,
        requests: Arc::clone(&requests),
    };
    let num_calls = responder.responses.len();
    Mock::given(method("POST"))
        .and(path_regex(
            ".*/models/gemini-3\\.5-flash:streamGenerateContent$",
        ))
        .respond_with(responder)
        .up_to_n_times(num_calls as u64)
        .expect(num_calls as u64)
        .mount(server)
        .await;

    GeminiRequestLog { requests }
}

pub(super) fn gemini_builder() -> TestCodexBuilder {
    test_codex().with_config(|config| {
        let mut provider = ModelProviderInfo::create_gemini_provider();
        provider.base_url = config.model_provider.base_url.clone();
        provider.experimental_bearer_token = Some("mock-gemini-key".to_string());
        config.model = Some(GEMINI_3_5_FLASH_MODEL.to_string());
        config.model_catalog = Some(gemini_model_catalog());
        config.model_provider_id = GEMINI_PROVIDER_ID.to_string();
        config.model_provider = provider;
    })
}

fn gemini_sse(chunks: Vec<Value>) -> String {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str("data: ");
        body.push_str(&chunk.to_string());
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    body
}

pub(super) fn gemini_function_call_sse(
    name: &str,
    args: Value,
    thought_signature: Option<&str>,
) -> String {
    gemini_function_calls_sse(vec![(name, args, thought_signature)])
}

fn gemini_function_calls_sse(calls: Vec<(&str, Value, Option<&str>)>) -> String {
    let parts = calls
        .into_iter()
        .map(|(name, args, thought_signature)| {
            let mut part = json!({
                "functionCall": {
                    "name": name,
                    "args": args,
                }
            });
            if let Some(thought_signature) = thought_signature {
                part["thoughtSignature"] = json!(thought_signature);
            }
            part
        })
        .collect::<Vec<_>>();

    gemini_sse(vec![json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": parts,
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 1,
            "thoughtsTokenCount": 1,
            "totalTokenCount": 12,
        },
    })])
}

pub(super) fn gemini_text_sse(text: &str) -> String {
    gemini_sse(vec![json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": text}],
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 1,
            "thoughtsTokenCount": 1,
            "totalTokenCount": 12,
        },
    })])
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_apply_patch_uses_exec_command_intercept() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let harness = TestCodexHarness::with_builder(gemini_builder()).await?;
    harness.write_file("gemini-edit.txt", "before\n").await?;
    harness
        .write_file("gemini-delete.txt", "delete me\n")
        .await?;

    let patch = "*** Begin Patch\n*** Add File: gemini-created.txt\n+created\n*** Update File: gemini-edit.txt\n@@\n-before\n+after\n*** Delete File: gemini-delete.txt\n*** End Patch";
    let command = format!("apply_patch <<'EOF'\n{patch}\nEOF\n");
    let requests = mount_gemini_sse_sequence(
        harness.server(),
        vec![
            gemini_function_call_sse(
                "exec_command",
                json!({
                    "cmd": command,
                    "yield_time_ms": 5000,
                }),
                Some("sig-exec"),
            ),
            gemini_text_sse("done"),
        ],
    )
    .await;

    harness
        .test()
        .submit_turn_with_permission_profile(
            "apply the patch through Gemini exec_command",
            PermissionProfile::Disabled,
        )
        .await?;

    assert_eq!(
        harness.read_file_text("gemini-created.txt").await?,
        "created\n"
    );
    assert_eq!(harness.read_file_text("gemini-edit.txt").await?, "after\n");
    assert!(!harness.path_exists("gemini-delete.txt").await?);

    let captured = requests.requests();
    assert_eq!(captured.len(), 2);
    let tool_names = captured[0]["tools"][0]["functionDeclarations"]
        .as_array()
        .expect("function declarations")
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .collect::<Vec<_>>();
    assert!(
        tool_names.contains(&"exec_command"),
        "Gemini request must expose exec_command: {tool_names:?}"
    );
    assert!(
        !tool_names.contains(&"apply_patch"),
        "Gemini request must not expose freeform apply_patch: {tool_names:?}"
    );

    let follow_up_parts = captured[1]["contents"]
        .as_array()
        .expect("follow-up contents")
        .iter()
        .flat_map(|content| content["parts"].as_array().expect("content parts").iter())
        .collect::<Vec<_>>();
    assert!(
        follow_up_parts
            .iter()
            .any(|part| { part["functionResponse"]["name"].as_str() == Some("exec_command") }),
        "Gemini follow-up must replay exec_command functionResponse: {follow_up_parts:?}"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn gemini_web_tools_execute_client_side_function_calls() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let harness = TestCodexHarness::with_builder(gemini_builder()).await?;
    let search_url = format!("{}/search", harness.server().uri());
    let _search_env = EnvVarGuard::set(WEB_SEARCH_BACKEND_ENV, search_url.as_ref());
    let fetch_url = format!("{}/page", harness.server().uri());

    Mock::given(method("GET"))
        .and(path("/search"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [{
                "title": "Gemini native port",
                "url": fetch_url,
                "content": "Phase 2 web search result",
            }]
        })))
        .expect(1)
        .mount(harness.server())
        .await;
    Mock::given(method("GET"))
        .and(path("/page"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html; charset=utf-8")
                .set_body_string("<html><body><h1>Fetched Gemini Page</h1></body></html>"),
        )
        .expect(1)
        .mount(harness.server())
        .await;

    let requests = mount_gemini_sse_sequence(
        harness.server(),
        vec![
            gemini_function_calls_sse(vec![
                (
                    "web_search",
                    json!({
                        "query": "gemini native phase 2",
                        "limit": 1,
                    }),
                    Some("sig-web"),
                ),
                (
                    "web_fetch",
                    json!({
                        "url": fetch_url,
                        "max_bytes": 2000,
                    }),
                    None,
                ),
            ]),
            gemini_text_sse("done"),
        ],
    )
    .await;

    harness
        .test()
        .submit_turn_with_permission_profile(
            "search and fetch through Gemini client-side web tools",
            PermissionProfile::Disabled,
        )
        .await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), 2);
    let tool_names = captured[0]["tools"][0]["functionDeclarations"]
        .as_array()
        .expect("function declarations")
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .collect::<Vec<_>>();
    assert!(
        tool_names.contains(&"web_search"),
        "Gemini request must expose client-side web_search: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"web_fetch"),
        "Gemini request must expose client-side web_fetch: {tool_names:?}"
    );

    let function_responses = captured[1]["contents"]
        .as_array()
        .expect("follow-up contents")
        .iter()
        .flat_map(|content| content["parts"].as_array().expect("content parts").iter())
        .filter_map(|part| part.get("functionResponse"))
        .collect::<Vec<_>>();
    let response_names = function_responses
        .iter()
        .filter_map(|response| response.get("name").and_then(Value::as_str))
        .collect::<Vec<_>>();
    assert_eq!(response_names, vec!["web_search", "web_fetch"]);
    assert!(
        function_responses[0]["response"]["output"]
            .as_str()
            .is_some_and(|output| output.contains("Phase 2 web search result")),
        "web_search functionResponse must include mocked search result: {function_responses:?}"
    );
    assert!(
        function_responses[1]["response"]["output"]
            .as_str()
            .is_some_and(|output| output.contains("Fetched Gemini Page")),
        "web_fetch functionResponse must include mocked fetched page: {function_responses:?}"
    );

    Ok(())
}
