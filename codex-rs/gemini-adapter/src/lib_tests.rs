use codex_api::ResponseEvent;
use codex_model_provider_info::ModelProviderInfo;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::protocol::MultiAgentVersion;
use codex_protocol::protocol::TokenUsage;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use http::StatusCode;
use pretty_assertions::assert_eq;
use reqwest::Response;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

use super::*;
use crate::GeminiToolChoice;

#[test]
fn gemini_catalog_uses_flat_multi_agent_v2_tools() {
    let model_info = model_config::gemini_model_catalog()
        .models
        .into_iter()
        .find(|model| model.slug == GEMINI_3_5_FLASH_MODEL)
        .expect("gemini flash model");

    assert_eq!(model_info.multi_agent_version, Some(MultiAgentVersion::V2));
}

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
    for request in &requests {
        assert!(
            request
                .headers
                .iter()
                .all(|(name, _)| !name.as_str().starts_with("x-codex-")),
            "native Gemini requests must not include x-codex-* headers"
        );
        let body: Value = request
            .body_json()
            .expect("Gemini request body should be JSON");
        assert!(body.get("prompt_cache_key").is_none());
        assert!(body.get("promptCacheKey").is_none());
    }

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
async fn mocked_gemini_per_day_quota_falls_back_to_pro() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(format!(
            "/models/{GEMINI_3_5_FLASH_MODEL}:streamGenerateContent"
        )))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "Quota exceeded.",
                "details": [{
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{
                        "quotaId": "GenerateRequestsPerDayPerProject"
                    }]
                }]
            }
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(format!(
            "/models/{GEMINI_3_1_PRO_PREVIEW_MODEL}:streamGenerateContent"
        )))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"done\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":1,\"totalTokenCount\":2}}\n\ndata: [DONE]\n\n",
            "text/event-stream",
        ))
        .expect(1)
        .mount(&server)
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
    let prompt = GeminiPrompt {
        instructions: "Reply done.".to_string(),
        input: vec![user_message("Reply done.")],
        tools: Vec::new(),
        output_schema: None,
        tool_choice: GeminiToolChoice::Auto,
    };

    let mut stream = stream_generate_content(
        reqwest::Client::new(),
        &provider,
        &model_info,
        prompt,
        Some(ReasoningEffort::Low),
    )
    .await
    .expect("fallback stream should start");
    while let Some(event) = stream.recv().await {
        if matches!(
            event.expect("mock fallback stream event"),
            ResponseEvent::Completed { .. }
        ) {
            break;
        }
    }

    let requests = server
        .received_requests()
        .await
        .expect("wiremock should record requests");
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0].url.path(),
        format!("/models/{GEMINI_3_5_FLASH_MODEL}:streamGenerateContent")
    );
    assert_eq!(
        requests[1].url.path(),
        format!("/models/{GEMINI_3_1_PRO_PREVIEW_MODEL}:streamGenerateContent")
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

#[tokio::test]
async fn live_gemini_accepts_sanitized_complex_schema_when_auth_is_set() {
    if !live_auth_configured() {
        eprintln!(
            "skipping live Gemini schema smoke test: configure GEMINI_API_KEY or Vertex ADC env vars"
        );
        return;
    }

    let provider = ModelProviderInfo::create_gemini_provider();
    let model_info = model_config::gemini_model_catalog()
        .models
        .into_iter()
        .find(|model| model.slug == GEMINI_3_5_FLASH_MODEL)
        .expect("gemini flash model");
    let prompt = GeminiPrompt {
        instructions: "You are running a schema acceptance smoke test. Reply with done."
            .to_string(),
        input: vec![user_message(
            "Do not call the tool. Reply with the word done.",
        )],
        tools: vec![complex_schema_tool()],
        output_schema: None,
        tool_choice: GeminiToolChoice::Auto,
    };
    let request = request_translator::build_generate_content_request(
        &prompt,
        &model_info,
        Some(ReasoningEffort::Low),
    )
    .expect("complex schema request should build");
    let request_value = serde_json::to_value(&request).expect("request JSON");
    let parameters = &request_value["tools"][0]["functionDeclarations"][0]["parameters"];
    assert!(parameters.get("additionalProperties").is_none());
    assert!(parameters["properties"]["metadata"].get("$ref").is_none());
    assert!(
        parameters["properties"]["metadata"]
            .get("additionalProperties")
            .is_none()
    );

    let response = send_live_generate_content(
        &provider,
        &model_info,
        &request,
        &request_value,
        "LIVE_GEMINI_SCHEMA",
    )
    .await;
    let schema_turn = collect_live_response(response, "LIVE_GEMINI_SCHEMA")
        .await
        .expect("live schema response should stream");
    assert!(
        schema_turn.token_usage.is_some(),
        "live schema response should map usageMetadata into TokenUsage"
    );
}

#[tokio::test]
async fn live_gemini_accepts_guardian_structured_output_when_auth_is_set() {
    if !live_auth_configured() {
        eprintln!(
            "skipping live Gemini guardian structured-output smoke test: configure GEMINI_API_KEY or Vertex ADC env vars"
        );
        return;
    }

    let provider = ModelProviderInfo::create_gemini_provider();
    let model_info = model_config::gemini_model_catalog()
        .models
        .into_iter()
        .find(|model| model.slug == GEMINI_3_5_FLASH_MODEL)
        .expect("gemini flash model");
    let prompt = GeminiPrompt {
        instructions: "You are running a guardian structured-output smoke test. Return only JSON."
            .to_string(),
        input: vec![user_message(
            "Return {\"outcome\":\"allow\"} for this low-risk request.",
        )],
        tools: Vec::new(),
        output_schema: Some(json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "risk_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"]
                },
                "user_authorization": {
                    "type": "string",
                    "enum": ["unknown", "low", "medium", "high"]
                },
                "outcome": {
                    "type": "string",
                    "enum": ["allow", "deny"]
                },
                "rationale": {
                    "type": "string"
                }
            },
            "required": ["outcome"]
        })),
        tool_choice: GeminiToolChoice::Auto,
    };
    let request = request_translator::build_generate_content_request(
        &prompt,
        &model_info,
        Some(ReasoningEffort::Low),
    )
    .expect("guardian structured request should build");
    let request_value = serde_json::to_value(&request).expect("request JSON");
    assert_eq!(
        request_value["generationConfig"]["responseMimeType"],
        json!("application/json")
    );
    assert!(
        request_value["generationConfig"]
            .get("responseSchema")
            .is_some()
    );
    assert!(request_value.get("tools").is_none());
    assert!(request_value.get("toolConfig").is_none());

    let response = send_live_generate_content(
        &provider,
        &model_info,
        &request,
        &request_value,
        "LIVE_GEMINI_GUARDIAN_SCHEMA",
    )
    .await;
    let structured_turn = collect_live_response(response, "LIVE_GEMINI_GUARDIAN_SCHEMA")
        .await
        .expect("live guardian structured response should stream");
    let parsed: Value =
        serde_json::from_str(&structured_turn.text).expect("guardian response should be JSON");
    assert_eq!(parsed["outcome"], json!("allow"));
}

#[tokio::test]
async fn live_gemini_parallel_calls_replay_first_signature_only_when_auth_is_set() {
    if !live_auth_configured() {
        eprintln!(
            "skipping live Gemini parallel smoke test: configure GEMINI_API_KEY or Vertex ADC env vars"
        );
        return;
    }

    let provider = ModelProviderInfo::create_gemini_provider();
    let model_info = model_config::gemini_model_catalog()
        .models
        .into_iter()
        .find(|model| model.slug == GEMINI_3_5_FLASH_MODEL)
        .expect("gemini flash model");
    let tools = vec![weather_tool()];
    let mut input = vec![user_message(
        "Call get_weather for Paris and London in the same model response. Emit exactly two get_weather function calls now: one with city Paris and one with city London. Do not wait for a tool result between them. Do not answer in text.",
    )];
    let first_prompt = GeminiPrompt {
        instructions: "You are running a parallel function-calling smoke test.".to_string(),
        input: input.clone(),
        tools: tools.clone(),
        output_schema: None,
        tool_choice: GeminiToolChoice::Auto,
    };
    let first_request = request_translator::build_generate_content_request(
        &first_prompt,
        &model_info,
        Some(ReasoningEffort::Low),
    )
    .expect("parallel request should build");
    let first_request_value = serde_json::to_value(&first_request).expect("request JSON");
    let first_response = send_live_generate_content(
        &provider,
        &model_info,
        &first_request,
        &first_request_value,
        "LIVE_GEMINI_PARALLEL_STEP_1",
    )
    .await;
    let first_turn = collect_live_response(first_response, "LIVE_GEMINI_PARALLEL_STEP_1")
        .await
        .expect("live parallel first response should stream");
    assert!(
        first_turn.token_usage.is_some(),
        "live parallel response should map usageMetadata into TokenUsage"
    );
    assert_eq!(first_turn.calls.len(), 2);
    assert!(function_call_signature(&first_turn.calls[0]).is_some());
    assert_eq!(function_call_signature(&first_turn.calls[1]), None);

    let first_call_id = function_call_id(&first_turn.calls[0]).to_string();
    let second_call_id = function_call_id(&first_turn.calls[1]).to_string();
    input.extend(first_turn.calls.clone());
    input.push(ResponseItem::FunctionCallOutput {
        call_id: first_call_id,
        output: FunctionCallOutputPayload::from_text("Paris weather: clear".to_string()),
    });
    input.push(ResponseItem::FunctionCallOutput {
        call_id: second_call_id,
        output: FunctionCallOutputPayload::from_text("London weather: cloudy".to_string()),
    });

    let follow_up_prompt = GeminiPrompt {
        instructions: "You are running a parallel function-calling smoke test.".to_string(),
        input,
        tools,
        output_schema: None,
        tool_choice: GeminiToolChoice::Auto,
    };
    let follow_up_request = request_translator::build_generate_content_request(
        &follow_up_prompt,
        &model_info,
        Some(ReasoningEffort::Low),
    )
    .expect("parallel follow-up request should build");
    let follow_up_value = serde_json::to_value(&follow_up_request).expect("request JSON");
    assert_parallel_replay_request(&follow_up_value);
    let follow_up_response = send_live_generate_content(
        &provider,
        &model_info,
        &follow_up_request,
        &follow_up_value,
        "LIVE_GEMINI_PARALLEL_STEP_2",
    )
    .await;
    let _ = collect_live_response(follow_up_response, "LIVE_GEMINI_PARALLEL_STEP_2")
        .await
        .expect("live parallel follow-up response should stream");
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

struct LiveGeminiResponse {
    calls: Vec<ResponseItem>,
    token_usage: Option<TokenUsage>,
    text: String,
}

async fn send_live_generate_content<T: Serialize + ?Sized>(
    provider: &ModelProviderInfo,
    model_info: &codex_protocol::openai_models::ModelInfo,
    request: &T,
    request_value: &Value,
    label: &str,
) -> Response {
    let mut auth = auth::resolve_auth(provider)
        .await
        .expect("live Gemini auth should resolve");
    let url = auth.endpoint(provider, &model_info.slug);
    eprintln!("{label}_ENDPOINT {url}");
    eprintln!(
        "{label}_REQUEST {}",
        serde_json::to_string_pretty(request_value).expect("pretty request JSON")
    );

    let client = reqwest::Client::new();
    let mut response = send_request(&client, &auth, provider, &model_info.slug, request)
        .await
        .expect("live Gemini request should send");
    if response.status() == StatusCode::UNAUTHORIZED
        && auth
            .refresh_vertex_adc_once()
            .await
            .expect("ADC refresh should not fail")
    {
        response = send_request(&client, &auth, provider, &model_info.slug, request)
            .await
            .expect("live Gemini retry should send");
    }
    ensure_live_success_with_label(response, &url, label).await
}

async fn collect_live_response(
    response: Response,
    label: &str,
) -> codex_protocol::error::Result<LiveGeminiResponse> {
    let mut accumulator = response_translator::StreamAccumulator::default();
    let mut events = response.bytes_stream().eventsource();
    let mut chunk_index = 0_usize;

    while let Some(event) = events.next().await {
        let event = event.expect("live Gemini SSE event");
        if event.data.trim() == "[DONE]" {
            eprintln!("{label}_RESPONSE_DONE [DONE]");
            break;
        }
        let chunk_value: Value = serde_json::from_str(&event.data)?;
        eprintln!(
            "{label}_RESPONSE_CHUNK_{chunk_index} {}",
            serde_json::to_string_pretty(&chunk_value)?
        );
        assert!(accumulator.process_event_data(&event.data)?.is_empty());
        chunk_index += 1;
        if accumulator.is_finished() {
            break;
        }
    }

    let mut calls = Vec::new();
    let mut token_usage = None;
    let mut text = String::new();
    for event in accumulator.finish()? {
        match event {
            ResponseEvent::OutputItemDone(item @ ResponseItem::FunctionCall { .. }) => {
                eprintln!(
                    "{label}_CODEX_FUNCTION_CALL {}",
                    serde_json::to_string_pretty(&item)?
                );
                calls.push(item);
            }
            ResponseEvent::OutputItemDone(ResponseItem::Message { content, .. }) => {
                for item in content {
                    match item {
                        ContentItem::InputText { text: item_text }
                        | ContentItem::OutputText { text: item_text } => {
                            text.push_str(&item_text);
                        }
                        ContentItem::InputImage { .. } => {}
                    }
                }
                eprintln!("{label}_CODEX_TEXT {text}");
            }
            ResponseEvent::Completed {
                token_usage: usage, ..
            } => {
                eprintln!(
                    "{label}_CODEX_TOKEN_USAGE {}",
                    serde_json::to_string_pretty(&usage)?
                );
                token_usage = usage;
            }
            _ => {}
        }
    }
    Ok(LiveGeminiResponse {
        calls,
        token_usage,
        text,
    })
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
        output_schema: None,
        tool_choice: GeminiToolChoice::Auto,
    };
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

    let response = send_live_generate_content(
        provider,
        model_info,
        &request,
        &request_value,
        &format!("LIVE_GEMINI_STEP_{expected_step}"),
    )
    .await;
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

async fn ensure_live_success_with_label(response: Response, url: &str, label: &str) -> Response {
    if response.status().is_success() {
        return response;
    }

    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    assert!(
        !(status == StatusCode::BAD_REQUEST && body.contains("missing thought_signature")),
        "live Gemini {label} hit missing thought_signature 400 at {url}: {body}"
    );
    panic!("live Gemini {label} failed with {status} at {url}: {body}");
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

fn assert_parallel_replay_request(request: &Value) {
    let model_parts = request["contents"][1]["parts"]
        .as_array()
        .expect("model function call parts");
    assert_eq!(model_parts.len(), 2);
    assert!(model_parts[0].get("functionCall").is_some());
    assert!(
        model_parts[0]
            .get("thoughtSignature")
            .and_then(Value::as_str)
            .is_some_and(|signature| !signature.is_empty())
    );
    assert!(model_parts[1].get("functionCall").is_some());
    assert!(model_parts[1].get("thoughtSignature").is_none());

    let response_parts = request["contents"][2]["parts"]
        .as_array()
        .expect("user function response parts");
    assert_eq!(response_parts.len(), 2);
    assert!(response_parts[0].get("functionResponse").is_some());
    assert!(response_parts[1].get("functionResponse").is_some());
}

fn function_call_signature(call: &ResponseItem) -> Option<&str> {
    let ResponseItem::FunctionCall {
        thought_signature, ..
    } = call
    else {
        panic!("expected function call");
    };
    thought_signature.as_deref()
}

fn function_call_id(call: &ResponseItem) -> &str {
    let ResponseItem::FunctionCall { call_id, .. } = call else {
        panic!("expected function call");
    };
    call_id
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
        output_schema: None,
        tool_choice: GeminiToolChoice::Auto,
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

fn weather_tool() -> ToolSpec {
    ToolSpec::Function(ResponsesApiTool {
        name: "get_weather".to_string(),
        description: "Get weather for a city.".to_string(),
        strict: false,
        defer_loading: None,
        parameters: JsonSchema::object(
            BTreeMap::from([(
                "city".to_string(),
                JsonSchema::string(Some("City name.".to_string())),
            )]),
            Some(vec!["city".to_string()]),
            None,
        ),
        output_schema: None,
    })
}

fn complex_schema_tool() -> ToolSpec {
    ToolSpec::Function(ResponsesApiTool {
        name: "complex_schema_tool".to_string(),
        description: "A tool with a schema that must be sanitized for Gemini.".to_string(),
        strict: false,
        defer_loading: None,
        parameters: serde_json::from_value(json!({
            "type": "object",
            "additionalProperties": false,
            "$defs": {
                "Metadata": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "format": "uri-reference"}
                    }
                }
            },
            "properties": {
                "query": {"type": "string", "format": "regex"},
                "metadata": {
                    "$ref": "#/$defs/Metadata",
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "source": {"type": "string", "format": "uri-reference"},
                        "nested": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "priority": {"type": "integer"},
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string", "format": "uuid"}
                                }
                            }
                        }
                    }
                }
            },
            "required": ["query"]
        }))
        .expect("complex schema should parse"),
        output_schema: None,
    })
}
