use codex_api::ResponseEvent;
use codex_model_provider_info::ModelProviderInfo;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::protocol::TokenUsage;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use http::StatusCode;
use pretty_assertions::assert_eq;
use reqwest::Response;
use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;

use super::*;

#[tokio::test]
async fn mocked_native_gemini_round_trips_function_call_signature() {
    let server = MockServer::start().await;
    mount_gemini_sse(
        &server,
        r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"record_step","args":{"step":1}},"thoughtSignature":"sig-step-1"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2,"thoughtsTokenCount":3,"totalTokenCount":9}}"#,
        2,
    )
    .await;

    let mut provider = ModelProviderInfo::create_gemini_provider();
    provider.base_url = Some(server.uri());
    provider.env_key = None;
    provider.experimental_bearer_token = Some("mock-gemini-key".to_string());
    let model_info = model_config::gemini_model_catalog()
        .models
        .into_iter()
        .find(|model| model.slug == GEMINI_3_5_FLASH_MODEL)
        .expect("gemini flash model");
    let tools = step_tool();
    let mut input = vec![user_message(
        "Call record_step once with step 1, then after the tool response call it with step 2.",
    )];

    let first_call = next_function_call(&provider, &model_info, input.clone(), tools.clone()).await;
    let first_call_id = match &first_call {
        ResponseItem::FunctionCall {
            call_id,
            thought_signature,
            ..
        } => {
            assert_eq!(thought_signature.as_deref(), Some("sig-step-1"));
            call_id.clone()
        }
        _ => panic!("expected first function call"),
    };
    input.push(first_call);
    input.push(ResponseItem::FunctionCallOutput {
        call_id: first_call_id,
        output: FunctionCallOutputPayload::from_text("recorded step 1".to_string()),
    });

    let second_call = next_function_call(&provider, &model_info, input, tools).await;
    assert!(matches!(
        second_call,
        ResponseItem::FunctionCall {
            thought_signature: Some(_),
            ..
        }
    ));

    let requests = server
        .received_requests()
        .await
        .expect("wiremock should record requests");
    assert_eq!(requests.len(), 2);
    let first_body: Value = requests[0]
        .body_json()
        .expect("first Gemini request body should be JSON");
    let second_body: Value = requests[1]
        .body_json()
        .expect("second Gemini request body should be JSON");

    assert_eq!(
        requests[0].url.path(),
        format!("/models/{GEMINI_3_5_FLASH_MODEL}:streamGenerateContent")
    );
    assert!(
        requests[0]
            .url
            .query_pairs()
            .any(|(name, value)| name == "alt" && value == "sse")
    );
    let has_api_key = requests[0]
        .url
        .query_pairs()
        .any(|(name, value)| name == "key" && value == "mock-gemini-key");
    let has_bearer = requests[0]
        .headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("Bearer "));
    assert!(
        has_api_key || has_bearer,
        "Gemini mock request must include either AI Studio key query auth or Vertex bearer auth"
    );
    assert_eq!(
        first_body["generationConfig"]["thinkingConfig"],
        json!({"thinkingLevel": "high", "includeThoughts": true})
    );
    assert!(
        first_body["generationConfig"]
            .get("thinkingBudget")
            .is_none()
    );
    assert_eq!(
        first_body["tools"][0]["functionDeclarations"][0]["name"],
        json!("record_step")
    );
    assert_eq!(
        second_body["contents"][1]["parts"][0]["functionCall"],
        json!({"name": "record_step", "args": {"step": 1}})
    );
    assert_eq!(
        second_body["contents"][1]["parts"][0]["thoughtSignature"],
        json!("sig-step-1")
    );
    assert_eq!(
        second_body["contents"][2]["parts"][0]["functionResponse"],
        json!({"name": "record_step", "response": {"output": "recorded step 1"}})
    );
}

#[tokio::test]
async fn live_gemini_three_step_tool_loop_when_auth_is_set() {
    if !live_auth_configured() {
        eprintln!(
            "skipping live Gemini smoke test: configure GEMINI_API_KEY or Vertex ADC env vars"
        );
        return;
    }

    let provider = ModelProviderInfo::create_gemini_provider();
    let model_info = model_config::gemini_model_catalog()
        .models
        .into_iter()
        .find(|model| model.slug == GEMINI_3_5_FLASH_MODEL)
        .expect("gemini flash model");
    let mut input = vec![ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: "Call record_step exactly once with {\"step\":1}. After each tool response, call record_step once with the next step until step 3, then answer done.".to_string(),
        }],
        phase: None,
    }];
    let tools = vec![ToolSpec::Function(ResponsesApiTool {
        name: "record_step".to_string(),
        description: "Record a numbered step.".to_string(),
        strict: false,
        defer_loading: None,
        parameters: JsonSchema::object(
            BTreeMap::from([(
                "step".to_string(),
                JsonSchema::integer(Some("Step number to record.".to_string())),
            )]),
            Some(vec!["step".to_string()]),
            None,
        ),
        output_schema: None,
    })];

    for expected_step in 1_usize..=3 {
        let turn = live_next_function_call(
            &provider,
            &model_info,
            input.clone(),
            tools.clone(),
            expected_step,
        )
        .await
        .expect("live Gemini tool turn should complete");
        assert_token_usage_matches_raw(&turn.raw_usage, &turn.token_usage);
        let call = turn.call;
        let ResponseItem::FunctionCall {
            call_id,
            arguments,
            thought_signature,
            ..
        } = &call
        else {
            unreachable!();
        };
        assert!(
            thought_signature
                .as_ref()
                .is_some_and(|sig| !sig.is_empty()),
            "Gemini function call for step {expected_step} must include thoughtSignature"
        );
        assert_eq!(
            serde_json::from_str::<Value>(arguments).expect("function call arguments"),
            json!({"step": expected_step})
        );
        let call_id = call_id.clone();

        input.push(call);
        input.push(ResponseItem::FunctionCallOutput {
            call_id,
            output: FunctionCallOutputPayload::from_text(format!("recorded step {expected_step}")),
        });
    }
}

#[test]
fn vertex_endpoint_uses_global_aiplatform_host_for_gemini_3_x() {
    let provider = ModelProviderInfo::create_gemini_provider();
    let auth = auth::GeminiAuth::VertexAdc {
        project: "test-project".to_string(),
        token: "test-token".to_string(),
    };

    assert_eq!(
        auth.endpoint(&provider, GEMINI_3_5_FLASH_MODEL),
        format!(
            "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/google/models/{GEMINI_3_5_FLASH_MODEL}:streamGenerateContent"
        )
    );
}

struct LiveGeminiTurn {
    call: ResponseItem,
    raw_usage: Value,
    token_usage: TokenUsage,
}

async fn live_next_function_call(
    provider: &ModelProviderInfo,
    model_info: &codex_protocol::openai_models::ModelInfo,
    input: Vec<ResponseItem>,
    tools: Vec<ToolSpec>,
    expected_step: usize,
) -> codex_protocol::error::Result<LiveGeminiTurn> {
    let prompt = GeminiPrompt {
        instructions: "You are running a smoke test. Use the function tool as instructed."
            .to_string(),
        input,
        tools,
    };
    let mut auth = auth::resolve_auth(provider).await?;
    let request = request_translator::build_generate_content_request(
        &prompt,
        model_info,
        Some(ReasoningEffort::Low),
    )?;
    let request_value = serde_json::to_value(&request)?;
    assert_eq!(
        request_value["generationConfig"]["thinkingConfig"],
        json!({"thinkingLevel": "low", "includeThoughts": true})
    );
    assert!(
        request_value["generationConfig"]
            .get("thinkingBudget")
            .is_none()
    );
    assert_signed_replay_shape(&request_value, expected_step);

    let url = auth.endpoint(provider, &model_info.slug);
    eprintln!("LIVE_GEMINI_STEP_{expected_step}_ENDPOINT {url}");
    eprintln!(
        "LIVE_GEMINI_STEP_{expected_step}_REQUEST {}",
        serde_json::to_string_pretty(&request_value)?
    );

    let client = reqwest::Client::new();
    let mut response = send_request(&client, &auth, provider, &model_info.slug, &request).await?;
    if response.status() == StatusCode::UNAUTHORIZED && auth.refresh_vertex_adc_once().await? {
        response = send_request(&client, &auth, provider, &model_info.slug, &request).await?;
    }
    let response = ensure_live_success(response, &url, expected_step).await;
    let mut accumulator = response_translator::StreamAccumulator::default();
    let mut events = response.bytes_stream().eventsource();
    let mut raw_usage = None;
    let mut chunk_index = 0_usize;

    while let Some(event) = events.next().await {
        let event = event.expect("live Gemini SSE event");
        if event.data.trim() == "[DONE]" {
            eprintln!("LIVE_GEMINI_STEP_{expected_step}_RESPONSE_DONE [DONE]");
            break;
        }
        let chunk_value: Value = serde_json::from_str(&event.data)?;
        if let Some(usage) = chunk_value.get("usageMetadata") {
            raw_usage = Some(usage.clone());
        }
        eprintln!(
            "LIVE_GEMINI_STEP_{expected_step}_RESPONSE_CHUNK_{chunk_index} {}",
            serde_json::to_string_pretty(&chunk_value)?
        );
        assert!(accumulator.process_event_data(&event.data)?.is_empty());
        chunk_index += 1;
        if accumulator.is_finished() {
            break;
        }
    }

    let mut call = None;
    let mut token_usage = None;
    for event in accumulator.finish()? {
        match event {
            ResponseEvent::OutputItemDone(item @ ResponseItem::FunctionCall { .. }) => {
                eprintln!(
                    "LIVE_GEMINI_STEP_{expected_step}_CODEX_FUNCTION_CALL {}",
                    serde_json::to_string_pretty(&item)?
                );
                call = Some(item);
            }
            ResponseEvent::Completed {
                token_usage: usage, ..
            } => {
                eprintln!(
                    "LIVE_GEMINI_STEP_{expected_step}_CODEX_TOKEN_USAGE {}",
                    serde_json::to_string_pretty(&usage)?
                );
                token_usage = usage;
            }
            _ => {}
        }
    }

    Ok(LiveGeminiTurn {
        call: call.expect("Gemini should produce a function call"),
        raw_usage: raw_usage.expect("live Gemini response should include usageMetadata"),
        token_usage: token_usage.expect("live Gemini completed event should include TokenUsage"),
    })
}

async fn ensure_live_success(response: Response, url: &str, expected_step: usize) -> Response {
    if response.status().is_success() {
        return response;
    }

    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    assert!(
        !(status == StatusCode::BAD_REQUEST && body.contains("missing thought_signature")),
        "live Gemini step {expected_step} hit missing thought_signature 400 at {url}: {body}"
    );
    panic!("live Gemini step {expected_step} failed with {status} at {url}: {body}");
}

fn live_auth_configured() -> bool {
    non_empty_env(GEMINI_API_KEY_ENV_VAR)
        || (vertex_live_env_enabled()
            && non_empty_env(GOOGLE_CLOUD_PROJECT_ENV_VAR)
            && non_empty_env(GOOGLE_CLOUD_LOCATION_ENV_VAR))
}

fn non_empty_env(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
}

fn vertex_live_env_enabled() -> bool {
    std::env::var(GOOGLE_GENAI_USE_VERTEXAI_ENV_VAR).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes"
        )
    })
}

fn assert_signed_replay_shape(request: &Value, expected_step: usize) {
    let mut signed_function_calls = 0_usize;
    let mut function_responses = 0_usize;
    for content in request["contents"].as_array().into_iter().flatten() {
        for part in content["parts"].as_array().into_iter().flatten() {
            if part.get("functionCall").is_some() {
                assert!(
                    part.get("thoughtSignature")
                        .and_then(Value::as_str)
                        .is_some_and(|signature| !signature.is_empty()),
                    "replayed functionCall must include thoughtSignature"
                );
                signed_function_calls += 1;
            }
            if part.get("functionResponse").is_some() {
                function_responses += 1;
            }
        }
    }
    assert_eq!(signed_function_calls, expected_step - 1);
    assert_eq!(function_responses, expected_step - 1);
}

fn assert_token_usage_matches_raw(raw_usage: &Value, token_usage: &TokenUsage) {
    let expected = TokenUsage {
        input_tokens: raw_usage
            .get("promptTokenCount")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        cached_input_tokens: raw_usage
            .get("cachedContentTokenCount")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        output_tokens: raw_usage
            .get("candidatesTokenCount")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        reasoning_output_tokens: raw_usage
            .get("thoughtsTokenCount")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        total_tokens: raw_usage
            .get("totalTokenCount")
            .and_then(Value::as_i64)
            .unwrap_or_else(|| {
                raw_usage
                    .get("promptTokenCount")
                    .and_then(Value::as_i64)
                    .unwrap_or_default()
                    + raw_usage
                        .get("candidatesTokenCount")
                        .and_then(Value::as_i64)
                        .unwrap_or_default()
                    + raw_usage
                        .get("thoughtsTokenCount")
                        .and_then(Value::as_i64)
                        .unwrap_or_default()
            }),
    };
    assert_eq!(token_usage, &expected);
}

async fn mount_gemini_sse(server: &MockServer, chunk: &str, expected_requests: u64) {
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            format!("data: {chunk}\n\ndata: [DONE]\n\n"),
            "text/event-stream",
        ))
        .expect(expected_requests)
        .mount(server)
        .await;
}

async fn next_function_call(
    provider: &ModelProviderInfo,
    model_info: &codex_protocol::openai_models::ModelInfo,
    input: Vec<ResponseItem>,
    tools: Vec<ToolSpec>,
) -> ResponseItem {
    let prompt = GeminiPrompt {
        instructions: "You are running a mock smoke test.".to_string(),
        input,
        tools,
    };
    let mut stream = stream_generate_content(
        reqwest::Client::new(),
        provider,
        model_info,
        prompt,
        Some(ReasoningEffort::High),
    )
    .await
    .expect("mock Gemini request should start");

    let mut call = None;
    while let Some(event) = stream.recv().await {
        match event.expect("mock Gemini stream event") {
            ResponseEvent::OutputItemDone(item @ ResponseItem::FunctionCall { .. }) => {
                call = Some(item);
            }
            ResponseEvent::OutputItemDone(_) => {}
            ResponseEvent::Completed { .. } => break,
            ResponseEvent::Created
            | ResponseEvent::OutputItemAdded(_)
            | ResponseEvent::ServerModel(_)
            | ResponseEvent::ModelVerifications(_)
            | ResponseEvent::ServerReasoningIncluded(_)
            | ResponseEvent::OutputTextDelta(_)
            | ResponseEvent::ToolCallInputDelta { .. }
            | ResponseEvent::ReasoningSummaryDelta { .. }
            | ResponseEvent::ReasoningContentDelta { .. }
            | ResponseEvent::ReasoningSummaryPartAdded { .. }
            | ResponseEvent::RateLimits(_)
            | ResponseEvent::ModelsEtag(_) => {}
        }
    }
    call.expect("Gemini should produce a function call")
}

fn user_message(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase: None,
    }
}

fn step_tool() -> Vec<ToolSpec> {
    vec![ToolSpec::Function(ResponsesApiTool {
        name: "record_step".to_string(),
        description: "Record a numbered step.".to_string(),
        strict: false,
        defer_loading: None,
        parameters: JsonSchema::object(
            BTreeMap::from([(
                "step".to_string(),
                JsonSchema::integer(Some("Step number to record.".to_string())),
            )]),
            Some(vec!["step".to_string()]),
            None,
        ),
        output_schema: None,
    })]
}
