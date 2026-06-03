use std::sync::OnceLock;
use std::time::Duration;
use std::time::Instant;

use codex_model_provider_info::ModelProviderInfo;
use codex_protocol::error::CodexErr;
use codex_protocol::error::EnvVarError;
use codex_protocol::error::Result;
use http::header::AUTHORIZATION;
use reqwest::RequestBuilder;
use tokio::process::Command;
use tokio::sync::Mutex;

pub const GEMINI_API_KEY_ENV_VAR: &str = "GEMINI_API_KEY";
pub const GOOGLE_GENAI_USE_VERTEXAI_ENV_VAR: &str = "GOOGLE_GENAI_USE_VERTEXAI";
pub const GOOGLE_CLOUD_PROJECT_ENV_VAR: &str = "GOOGLE_CLOUD_PROJECT";
pub const GOOGLE_CLOUD_LOCATION_ENV_VAR: &str = "GOOGLE_CLOUD_LOCATION";

const AI_STUDIO_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const ADC_REFRESH_WINDOW: Duration = Duration::from_secs(55 * 60);

static ADC_TOKEN: OnceLock<Mutex<Option<CachedToken>>> = OnceLock::new();

#[derive(Debug, Clone)]
pub(crate) enum GeminiAuth {
    AiStudio { api_key: String },
    VertexAdc { project: String, token: String },
}

#[derive(Debug, Clone)]
struct CachedToken {
    token: String,
    fetched_at: Instant,
}

impl GeminiAuth {
    pub(crate) fn endpoint(&self, provider: &ModelProviderInfo, model: &str) -> String {
        if let Some(base_url) = provider.base_url.as_deref() {
            return format!(
                "{}/models/{model}:streamGenerateContent",
                base_url.trim_end_matches('/')
            );
        }

        match self {
            Self::AiStudio { .. } => {
                format!("{AI_STUDIO_BASE_URL}/models/{model}:streamGenerateContent")
            }
            Self::VertexAdc { project, .. } => format!(
                "https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/google/models/{model}:streamGenerateContent"
            ),
        }
    }

    pub(crate) fn apply_to_request(&self, builder: RequestBuilder) -> RequestBuilder {
        match self {
            Self::AiStudio { api_key } => builder.query(&[("key", api_key.as_str())]),
            Self::VertexAdc { token, .. } => {
                builder.header(AUTHORIZATION, format!("Bearer {token}"))
            }
        }
    }

    pub(crate) async fn refresh_vertex_adc_once(&mut self) -> Result<bool> {
        match self {
            Self::AiStudio { .. } => Ok(false),
            Self::VertexAdc { token, .. } => {
                *token = refresh_adc_token().await?;
                Ok(true)
            }
        }
    }
}

pub(crate) async fn resolve_auth(provider: &ModelProviderInfo) -> Result<GeminiAuth> {
    if use_vertex() {
        let project = env_var(GOOGLE_CLOUD_PROJECT_ENV_VAR)?;
        let _location = env_var(GOOGLE_CLOUD_LOCATION_ENV_VAR)?;
        let token = cached_adc_token().await?;
        return Ok(GeminiAuth::VertexAdc { project, token });
    }

    let api_key = provider
        .api_key()?
        .or_else(|| provider.experimental_bearer_token.clone())
        .or_else(|| std::env::var(GEMINI_API_KEY_ENV_VAR).ok())
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            CodexErr::EnvVar(EnvVarError {
                var: GEMINI_API_KEY_ENV_VAR.to_string(),
                instructions: Some("Set GEMINI_API_KEY to use Gemini AI Studio.".to_string()),
            })
        })?;
    Ok(GeminiAuth::AiStudio { api_key })
}

fn use_vertex() -> bool {
    std::env::var(GOOGLE_GENAI_USE_VERTEXAI_ENV_VAR).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes"
        )
    })
}

fn env_var(name: &str) -> Result<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            CodexErr::EnvVar(EnvVarError {
                var: name.to_string(),
                instructions: None,
            })
        })
}

async fn cached_adc_token() -> Result<String> {
    let cache = ADC_TOKEN.get_or_init(|| Mutex::new(None));
    {
        let guard = cache.lock().await;
        if let Some(token) = guard.as_ref()
            && token.fetched_at.elapsed() < ADC_REFRESH_WINDOW
        {
            return Ok(token.token.clone());
        }
    }

    refresh_adc_token().await
}

async fn refresh_adc_token() -> Result<String> {
    let token = fetch_adc_token().await?;
    let cache = ADC_TOKEN.get_or_init(|| Mutex::new(None));
    let mut guard = cache.lock().await;
    *guard = Some(CachedToken {
        token: token.clone(),
        fetched_at: Instant::now(),
    });
    Ok(token)
}

async fn fetch_adc_token() -> Result<String> {
    let output = Command::new("gcloud")
        .args(["auth", "application-default", "print-access-token"])
        .output()
        .await?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CodexErr::InvalidRequest(format!(
            "failed to get Vertex ADC token from gcloud: {stderr}"
        )));
    }
    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if token.is_empty() {
        return Err(CodexErr::InvalidRequest(
            "gcloud returned an empty Vertex ADC token".to_string(),
        ));
    }
    Ok(token)
}
