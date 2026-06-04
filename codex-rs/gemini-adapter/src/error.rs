use codex_protocol::error::CodexErr;
use codex_protocol::error::ConnectionFailedError;
use codex_protocol::error::ResponseStreamFailed;
use codex_protocol::error::UnexpectedResponseError;
use http::StatusCode;
use serde::Deserialize;
use std::time::Duration;

const SHORT_RETRY_DELAY: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GeminiErrorDecision {
    RetrySameModel(Option<Duration>),
    FallbackModel,
    ContextWindowExceeded,
    NoRetry,
}

#[derive(Debug, Deserialize)]
struct GoogleRpcBody {
    error: Option<GoogleRpcError>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleRpcError {
    status: Option<String>,
    message: Option<String>,
    #[serde(default)]
    details: Vec<GoogleRpcDetail>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleRpcDetail {
    #[serde(rename = "@type")]
    type_url: Option<String>,
    retry_delay: Option<String>,
    #[serde(default)]
    violations: Vec<QuotaViolation>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct QuotaViolation {
    quota_id: Option<String>,
}

pub(crate) fn request(err: reqwest::Error) -> CodexErr {
    if err.is_timeout() {
        CodexErr::RequestTimeout
    } else if err.is_connect() {
        CodexErr::ConnectionFailed(ConnectionFailedError { source: err })
    } else {
        CodexErr::ResponseStreamFailed(ResponseStreamFailed {
            source: err,
            request_id: None,
        })
    }
}

pub(crate) fn codex_error_for_status_body(status: StatusCode, body: String, url: &str) -> CodexErr {
    if classify_google_rpc_error(status, &body) == GeminiErrorDecision::ContextWindowExceeded {
        return CodexErr::ContextWindowExceeded;
    }

    CodexErr::UnexpectedStatus(UnexpectedResponseError {
        status,
        body,
        url: Some(url.to_string()),
        cf_ray: None,
        request_id: None,
        identity_authorization_error: None,
        identity_error_code: None,
    })
}

pub(crate) fn classify_google_rpc_error(status: StatusCode, body: &str) -> GeminiErrorDecision {
    if is_context_window_error(status, body) {
        return GeminiErrorDecision::ContextWindowExceeded;
    }

    if status != StatusCode::TOO_MANY_REQUESTS {
        return GeminiErrorDecision::NoRetry;
    }

    let Ok(parsed) = serde_json::from_str::<GoogleRpcBody>(body) else {
        return GeminiErrorDecision::NoRetry;
    };
    let Some(error) = parsed.error else {
        return GeminiErrorDecision::NoRetry;
    };

    let mut saw_per_minute = false;
    let mut saw_per_day = false;
    let mut retry_delay = None;

    for detail in error.details {
        if detail
            .type_url
            .as_deref()
            .is_some_and(|type_url| type_url.ends_with("google.rpc.RetryInfo"))
            && let Some(delay) = detail.retry_delay.as_deref().and_then(parse_retry_delay)
        {
            retry_delay = Some(delay);
        }

        if detail
            .type_url
            .as_deref()
            .is_some_and(|type_url| type_url.ends_with("google.rpc.QuotaFailure"))
        {
            for violation in detail.violations {
                let Some(quota_id) = violation.quota_id else {
                    continue;
                };
                saw_per_minute |= quota_id.contains("PerMinute");
                saw_per_day |= quota_id.contains("PerDay");
            }
        }
    }

    if saw_per_day {
        return GeminiErrorDecision::FallbackModel;
    }

    if let Some(delay) = retry_delay {
        if delay <= SHORT_RETRY_DELAY {
            return GeminiErrorDecision::RetrySameModel(Some(delay));
        }
        return GeminiErrorDecision::FallbackModel;
    }

    if saw_per_minute {
        return GeminiErrorDecision::RetrySameModel(None);
    }

    let status_text = error.status.unwrap_or_default();
    if status_text == "RESOURCE_EXHAUSTED" {
        return GeminiErrorDecision::FallbackModel;
    }

    GeminiErrorDecision::NoRetry
}

fn parse_retry_delay(value: &str) -> Option<Duration> {
    let seconds = value.strip_suffix('s')?;
    if seconds.is_empty() || seconds.starts_with('-') {
        return None;
    }
    let (whole, fractional) = seconds.split_once('.').unwrap_or((seconds, ""));
    let secs = whole.parse::<u64>().ok()?;
    let mut nanos = 0_u32;
    if !fractional.is_empty() {
        let mut padded = fractional.chars().take(9).collect::<String>();
        while padded.len() < 9 {
            padded.push('0');
        }
        nanos = padded.parse::<u32>().ok()?;
    }
    Some(Duration::new(secs, nanos))
}

fn is_context_window_error(status: StatusCode, body: &str) -> bool {
    if status != StatusCode::BAD_REQUEST && status != StatusCode::PAYLOAD_TOO_LARGE {
        return false;
    }

    let message = serde_json::from_str::<GoogleRpcBody>(body)
        .ok()
        .and_then(|body| body.error)
        .and_then(|error| error.message)
        .unwrap_or_else(|| body.to_string());
    let message = message.to_ascii_lowercase();
    (message.contains("context") && (message.contains("window") || message.contains("length")))
        || (message.contains("token") && message.contains("limit"))
}

#[cfg(test)]
#[path = "error_tests.rs"]
mod tests;
