mod auth;
mod error;
pub mod model_config;
mod request_translator;
mod response_translator;
mod schema_sanitizer;
mod signature_store;
mod tool_translator;

use codex_api::ResponseEvent;
use codex_model_provider_info::ModelProviderInfo;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ReasoningEffort;
use codex_tools::ToolSpec;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use http::StatusCode;
use reqwest::Client;
use reqwest::Response;
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::warn;

pub use auth::GEMINI_API_KEY_ENV_VAR;
pub use auth::GOOGLE_CLOUD_LOCATION_ENV_VAR;
pub use auth::GOOGLE_CLOUD_PROJECT_ENV_VAR;
pub use auth::GOOGLE_GENAI_USE_VERTEXAI_ENV_VAR;
pub use model_config::GEMINI_3_1_PRO_PREVIEW_MODEL;
pub use model_config::GEMINI_3_5_FLASH_MODEL;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeminiToolChoice {
    Auto,
    Any { allowed_function_names: Vec<String> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeminiThoughtSummaryDisplay {
    Hidden,
    Visible,
}

/// Phase-1 Gemini prompt data, kept independent from codex-core's Prompt type.
#[derive(Debug, Clone)]
pub struct GeminiPrompt {
    pub instructions: String,
    pub input: Vec<ResponseItem>,
    pub tools: Vec<ToolSpec>,
    pub output_schema: Option<Value>,
    pub tool_choice: GeminiToolChoice,
    pub thought_summary_display: GeminiThoughtSummaryDisplay,
}

pub async fn stream_generate_content(
    client: Client,
    provider: &ModelProviderInfo,
    model_info: &ModelInfo,
    prompt: GeminiPrompt,
    effort: Option<ReasoningEffort>,
) -> Result<mpsc::Receiver<Result<ResponseEvent>>> {
    let mut auth = auth::resolve_auth(provider).await?;
    let request = request_translator::build_generate_content_request(&prompt, model_info, effort)?;
    let response =
        send_request_with_retries(&client, &mut auth, provider, &model_info.slug, &request).await?;

    let (tx, rx) = mpsc::channel(1600);
    tokio::spawn(async move {
        let mut accumulator =
            response_translator::StreamAccumulator::new(prompt.thought_summary_display);
        let mut events = response.bytes_stream().eventsource();
        while let Some(event) = events.next().await {
            let event = match event {
                Ok(event) => event,
                Err(err) => {
                    let _ = tx.send(Err(CodexErr::Stream(err.to_string(), None))).await;
                    return;
                }
            };
            if event.data.trim() == "[DONE]" {
                break;
            }
            match accumulator.process_event_data(&event.data) {
                Ok(items) => {
                    for item in items {
                        if tx.send(Ok(item)).await.is_err() {
                            return;
                        }
                    }
                }
                Err(err) => {
                    let _ = tx.send(Err(err)).await;
                    return;
                }
            }
            if accumulator.is_finished() {
                break;
            }
        }
        match accumulator.finish() {
            Ok(items) => {
                for item in items {
                    if tx.send(Ok(item)).await.is_err() {
                        return;
                    }
                }
            }
            Err(err) => {
                let _ = tx.send(Err(err)).await;
            }
        }
    });

    Ok(rx)
}

async fn send_request<T: serde::Serialize + ?Sized>(
    client: &Client,
    auth: &auth::GeminiAuth,
    provider: &ModelProviderInfo,
    model: &str,
    request: &T,
) -> Result<Response> {
    let url = auth.endpoint(provider, model);
    let builder = client.post(url).query(&[("alt", "sse")]);
    let builder = auth.apply_to_request(builder);
    builder.json(request).send().await.map_err(error::request)
}

async fn send_request_with_retries<T: serde::Serialize + ?Sized>(
    client: &Client,
    auth: &mut auth::GeminiAuth,
    provider: &ModelProviderInfo,
    model: &str,
    request: &T,
) -> Result<Response> {
    let mut active_model = model.to_string();
    let mut same_model_retries = 0_usize;
    let mut fallback_used = false;

    loop {
        let mut response = send_request(client, auth, provider, &active_model, request).await?;
        if response.status() == StatusCode::UNAUTHORIZED && auth.refresh_vertex_adc_once().await? {
            response = send_request(client, auth, provider, &active_model, request).await?;
        }

        if response.status().is_success() {
            return Ok(response);
        }

        let status = response.status();
        let url = auth.endpoint(provider, &active_model);
        let body = response.text().await.unwrap_or_default();
        match error::classify_google_rpc_error(status, &body) {
            error::GeminiErrorDecision::RetrySameModel(delay) if same_model_retries < 2 => {
                same_model_retries += 1;
                if let Some(delay) = delay {
                    tokio::time::sleep(delay).await;
                }
            }
            error::GeminiErrorDecision::FallbackModel if !fallback_used => {
                let Some(fallback_model) = fallback_model_for(&active_model) else {
                    return Err(error::codex_error_for_status_body(status, body, &url));
                };
                warn!(
                    model = %active_model,
                    fallback_model,
                    "Gemini quota response triggered model fallback"
                );
                active_model = fallback_model.to_string();
                same_model_retries = 0;
                fallback_used = true;
            }
            error::GeminiErrorDecision::ContextWindowExceeded
            | error::GeminiErrorDecision::NoRetry
            | error::GeminiErrorDecision::RetrySameModel(_)
            | error::GeminiErrorDecision::FallbackModel => {
                return Err(error::codex_error_for_status_body(status, body, &url));
            }
        }
    }
}

fn fallback_model_for(model: &str) -> Option<&'static str> {
    match model {
        GEMINI_3_5_FLASH_MODEL => Some(GEMINI_3_1_PRO_PREVIEW_MODEL),
        GEMINI_3_1_PRO_PREVIEW_MODEL => Some(GEMINI_3_5_FLASH_MODEL),
        _ => None,
    }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
