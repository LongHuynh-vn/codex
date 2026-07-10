#![allow(clippy::expect_used)]

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::Result;
use codex_gemini_adapter::GEMINI_3_5_FLASH_MODEL;
use codex_gemini_adapter::model_config::gemini_model_catalog;
use codex_model_provider_info::GEMINI_PROVIDER_ID;
use codex_model_provider_info::ModelProviderInfo;
use codex_protocol::config_types::WebSearchMode;
use codex_protocol::models::PermissionProfile;
use codex_protocol::protocol::GeminiSearchMode;
use codex_protocol::protocol::ThreadSettingsOverrides;
use core_test_support::skip_if_no_network;
use core_test_support::submit_thread_settings;
use core_test_support::test_codex::TestCodexBuilder;
use core_test_support::test_codex::TestCodexHarness;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use regex_lite::Regex;
use serde_json::Value;
use serde_json::json;
use serial_test::serial;
use wiremock::Match;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::Respond;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;
use wiremock::matchers::path_regex;

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
    response_delay: Option<Duration>,
}

struct GeminiRequestMatcher<F>(F);

impl<F> Match for GeminiRequestMatcher<F>
where
    F: Fn(&wiremock::Request) -> bool + Send + Sync,
{
    fn matches(&self, request: &wiremock::Request) -> bool {
        (self.0)(request)
    }
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
        let response = ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(body.clone());
        match self.response_delay {
            Some(delay) => response.set_delay(delay),
            None => response,
        }
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
        response_delay: None,
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

pub(super) async fn mount_gemini_sse_once_match<F>(
    server: &MockServer,
    matcher: F,
    response: String,
    response_delay: Option<Duration>,
) -> GeminiRequestLog
where
    F: Fn(&wiremock::Request) -> bool + Send + Sync + 'static,
{
    let requests = Arc::new(Mutex::new(Vec::new()));
    let responder = GeminiSseResponder {
        num_calls: AtomicUsize::new(0),
        responses: vec![response],
        requests: Arc::clone(&requests),
        response_delay,
    };
    Mock::given(method("POST"))
        .and(path_regex(
            ".*/models/gemini-3\\.5-flash:streamGenerateContent$",
        ))
        .and(GeminiRequestMatcher(matcher))
        .respond_with(responder)
        .up_to_n_times(1)
        .expect(1)
        .mount(server)
        .await;

    GeminiRequestLog { requests }
}

pub(super) fn gemini_builder() -> TestCodexBuilder {
    test_codex().with_config(|config| {
        let mut provider = ModelProviderInfo::create_gemini_provider();
        provider.base_url = config.model_provider.base_url.clone();
        // Authenticate via the mock bearer token rather than the GEMINI_API_KEY env var, so the
        // suite runs without that variable set (the integration gate unsets it). `api_key()` hard
        // errors when `env_key` is configured but the env var is missing, which short-circuits the
        // `experimental_bearer_token` fallback in `resolve_auth`; null it like the adapter's own
        // bearer-token unit test does.
        provider.env_key = None;
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

pub(super) fn gemini_empty_sse() -> String {
    gemini_sse(vec![json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [],
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 0,
            "thoughtsTokenCount": 0,
            "totalTokenCount": 10,
        },
    })])
}

fn gemini_function_declaration_names(request: &Value) -> Vec<&str> {
    request["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|tool| tool.get("functionDeclarations").and_then(Value::as_array))
        .flatten()
        .filter_map(|declaration| declaration.get("name").and_then(Value::as_str))
        .collect()
}

fn gemini_request_has_google_search(request: &Value) -> bool {
    request["tools"]
        .as_array()
        .is_some_and(|tools| tools.iter().any(|tool| tool.get("googleSearch").is_some()))
}

fn gemini_system_instruction(request: &Value) -> &str {
    request["systemInstruction"]["parts"][0]["text"]
        .as_str()
        .expect("Gemini request systemInstruction text")
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
    let instructions = captured[0]["systemInstruction"]["parts"][0]["text"]
        .as_str()
        .expect("Gemini request systemInstruction text");
    assert!(
        instructions.contains(
            "Use the `apply_patch` shell command for EVERY file creation and modification"
        ) && instructions.contains("regardless of file size or file count")
            && instructions.contains("Use `*** Add File` for new files, including long files")
            && instructions.contains(
                "fix the patch and call `apply_patch` again; never fall back to a shell write"
            ),
        "Gemini systemInstruction must require apply_patch for every file write: {instructions}"
    );
    assert!(
        instructions.contains("apply_patch <<'PATCH'"),
        "Gemini systemInstruction must show heredoc invocation: {instructions}"
    );
    assert!(
        instructions.contains("*** Begin Patch")
            && instructions.contains("*** Add File: <path>")
            && instructions.contains("*** Update File: <path>")
            && instructions.contains("*** Delete File: <path>")
            && instructions.contains("*** Move to: <new path>")
            && instructions.contains("*** End Patch"),
        "Gemini systemInstruction must preserve apply_patch envelope grammar: {instructions}"
    );
    assert!(
        instructions.contains("`cat`")
            && instructions.contains(
                "shell redirection or a heredoc other than the `apply_patch` invocation shown below"
            )
            && instructions.contains("`echo`")
            && instructions.contains("`printf`")
            && instructions.contains("`tee`")
            && instructions.contains("`python -c`")
            && instructions.contains("`sed -i`"),
        "Gemini systemInstruction must prohibit shell file writes: {instructions}"
    );
    assert!(
        instructions.contains("After creating or editing files"),
        "Gemini systemInstruction must include output conciseness marker: {instructions}"
    );
    assert_eq!(
        instructions
            .matches("After creating or editing files")
            .count(),
        1,
        "Gemini systemInstruction must include output conciseness marker once: {instructions}"
    );
    assert!(
        instructions.contains("do not reproduce the full file contents")
            && instructions.contains("unless the user explicitly asks"),
        "Gemini systemInstruction must include output conciseness guidance: {instructions}"
    );
    assert!(
        instructions.contains("Today's date is"),
        "Gemini systemInstruction must include current-date marker: {instructions}"
    );
    assert_eq!(
        instructions.matches("Today's date is").count(),
        1,
        "Gemini systemInstruction must include current-date marker once: {instructions}"
    );
    assert!(
        Regex::new(r"Today's date is \d{4}-\d{2}-\d{2} \([A-Za-z]+\)\.")
            .expect("date regex should compile")
            .is_match(instructions),
        "Gemini systemInstruction must include date-shaped current date: {instructions}"
    );
    assert!(
        instructions.contains(
            "For any time-sensitive query, you MUST use this as the current date, and trust web_search/web_fetch results over your training data when they conflict on dates or latest versions."
        ),
        "Gemini systemInstruction must include time-sensitive query guidance: {instructions}"
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
async fn gemini_search_modes_control_tools_grounding_and_hybrid_nudge() -> Result<()> {
    let cases = [
        (GeminiSearchMode::Tavily, true, false, false),
        (GeminiSearchMode::Grounding, false, true, false),
        (GeminiSearchMode::Hybrid, true, true, true),
        (GeminiSearchMode::Off, false, false, false),
    ];

    for (mode, expect_client_web_tools, expect_google_search, expect_hybrid_nudge) in cases {
        let harness = TestCodexHarness::with_builder(gemini_builder()).await?;
        submit_thread_settings(
            harness.test().codex.as_ref(),
            ThreadSettingsOverrides {
                gemini_search_mode: Some(mode),
                ..Default::default()
            },
        )
        .await?;
        let requests =
            mount_gemini_sse_sequence(harness.server(), vec![gemini_text_sse("done")]).await;

        harness
            .test()
            .submit_turn_with_permission_profile(
                "answer with the configured Gemini search mode",
                PermissionProfile::Disabled,
            )
            .await?;

        let captured = requests.requests();
        assert_eq!(captured.len(), 1);
        let tool_names = gemini_function_declaration_names(&captured[0]);
        assert_eq!(
            tool_names.contains(&"web_search"),
            expect_client_web_tools,
            "mode {mode:?} web_search mismatch: {tool_names:?}"
        );
        assert_eq!(
            tool_names.contains(&"web_fetch"),
            expect_client_web_tools,
            "mode {mode:?} web_fetch mismatch: {tool_names:?}"
        );
        assert_eq!(
            gemini_request_has_google_search(&captured[0]),
            expect_google_search,
            "mode {mode:?} googleSearch mismatch: {captured:?}"
        );
        let instructions = gemini_system_instruction(&captured[0]);
        assert_eq!(
            instructions.contains("Gemini search mode: Hybrid"),
            expect_hybrid_nudge,
            "mode {mode:?} hybrid nudge mismatch: {instructions}"
        );
        assert_eq!(
            instructions.matches("Gemini search mode: Hybrid").count(),
            usize::from(expect_hybrid_nudge),
            "mode {mode:?} hybrid nudge duplication mismatch: {instructions}"
        );
        if expect_hybrid_nudge {
            assert!(
                instructions.ends_with(
                    "If the user asks not to search, or freshness is unnecessary, skip search."
                ),
                "Hybrid nudge should be last in Gemini instructions: {instructions}"
            );
        }
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn gemini_web_tools_execute_client_side_function_calls() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let harness = TestCodexHarness::with_builder(gemini_builder()).await?;
    let fetch_url = format!("{}/page", harness.server().uri());

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
            gemini_function_calls_sse(vec![(
                "web_fetch",
                json!({
                    "url": fetch_url,
                    "max_bytes": 2000,
                }),
                None,
            )]),
            gemini_text_sse("done"),
        ],
    )
    .await;

    harness
        .test()
        .submit_turn_with_permission_profile(
            "fetch through Gemini client-side web tools",
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
    let web_fetch_description = captured[0]["tools"][0]["functionDeclarations"]
        .as_array()
        .expect("function declarations")
        .iter()
        .find(|tool| tool.get("name").and_then(Value::as_str) == Some("web_fetch"))
        .and_then(|tool| tool.get("description").and_then(Value::as_str))
        .expect("web_fetch declaration description");
    assert!(
        web_fetch_description.contains("web_search"),
        "web_fetch description must delegate from web_search: {web_fetch_description}"
    );
    let instructions = captured[0]["systemInstruction"]["parts"][0]["text"]
        .as_str()
        .expect("Gemini request systemInstruction text");
    assert!(
        instructions.contains("Research diligence"),
        "Gemini systemInstruction must include research diligence marker: {instructions}"
    );
    assert_eq!(
        instructions.matches("Research diligence").count(),
        1,
        "Gemini systemInstruction must include research diligence marker once: {instructions}"
    );
    assert!(
        instructions.contains("wrong source or entity")
            && instructions.contains("not reported")
            && instructions.contains("unavailable rather than guessing"),
        "Gemini systemInstruction must include research grounding guidance: {instructions}"
    );
    assert!(
        instructions.contains("call that exact tool")
            && instructions.contains("actually called web_fetch"),
        "Gemini systemInstruction must include tool compliance and honesty guidance: {instructions}"
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
    assert_eq!(response_names, vec!["web_fetch"]);
    assert!(
        function_responses[0]["response"]["output"]
            .as_str()
            .is_some_and(|output| output.contains("Fetched Gemini Page")),
        "web_fetch functionResponse must include mocked fetched page: {function_responses:?}"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_research_diligence_absent_when_web_tools_disabled() -> Result<()> {
    let harness = TestCodexHarness::with_builder(gemini_builder().with_config(|config| {
        config
            .web_search_mode
            .set(WebSearchMode::Disabled)
            .expect("test web_search_mode should satisfy constraints");
    }))
    .await?;

    let requests = mount_gemini_sse_sequence(harness.server(), vec![gemini_text_sse("done")]).await;

    harness
        .test()
        .submit_turn_with_permission_profile(
            "answer without Gemini client-side web tools",
            PermissionProfile::Disabled,
        )
        .await?;

    let captured = requests.requests();
    assert_eq!(captured.len(), 1);
    let tool_names = captured[0]["tools"][0]["functionDeclarations"]
        .as_array()
        .expect("function declarations")
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .collect::<Vec<_>>();
    assert!(
        !tool_names.contains(&"web_search"),
        "Gemini request must not expose client-side web_search when disabled: {tool_names:?}"
    );
    assert!(
        !tool_names.contains(&"web_fetch"),
        "Gemini request must not expose client-side web_fetch when disabled: {tool_names:?}"
    );
    let instructions = captured[0]["systemInstruction"]["parts"][0]["text"]
        .as_str()
        .expect("Gemini request systemInstruction text");
    assert!(
        !instructions.contains("Research diligence"),
        "Gemini systemInstruction must not include research diligence marker without web tools: {instructions}"
    );

    Ok(())
}
