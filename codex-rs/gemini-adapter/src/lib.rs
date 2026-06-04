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
use tokio::sync::mpsc;

pub use auth::GEMINI_API_KEY_ENV_VAR;
pub use auth::GOOGLE_CLOUD_LOCATION_ENV_VAR;
pub use auth::GOOGLE_CLOUD_PROJECT_ENV_VAR;
pub use auth::GOOGLE_GENAI_USE_VERTEXAI_ENV_VAR;
pub use model_config::GEMINI_3_1_PRO_PREVIEW_MODEL;
pub use model_config::GEMINI_3_5_FLASH_MODEL;

/// Phase-1 Gemini prompt data, kept independent from codex-core's Prompt type.
#[derive(Debug, Clone)]
pub struct GeminiPrompt {
    pub instructions: String,
    pub input: Vec<ResponseItem>,
    pub tools: Vec<ToolSpec>,
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
    let mut response = send_request(&client, &auth, provider, &model_info.slug, &request).await?;
    if response.status() == StatusCode::UNAUTHORIZED && auth.refresh_vertex_adc_once().await? {
        response = send_request(&client, &auth, provider, &model_info.slug, &request).await?;
    }
    let url = auth.endpoint(provider, &model_info.slug);
    let response = error::ensure_success(response, &url).await?;

    let (tx, rx) = mpsc::channel(1600);
    tokio::spawn(async move {
        let mut accumulator = response_translator::StreamAccumulator::default();
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

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
