use super::*;
use pretty_assertions::assert_eq;
use serde_json::json;

#[test]
fn retry_info_short_delay_silently_retries_same_model() {
    let body = rpc_error_body(vec![json!({
        "@type": "type.googleapis.com/google.rpc.RetryInfo",
        "retryDelay": "12.500s"
    })]);

    assert_eq!(
        classify_google_rpc_error(StatusCode::TOO_MANY_REQUESTS, &body),
        GeminiErrorDecision::RetrySameModel(Some(Duration::new(12, 500_000_000)))
    );
}

#[test]
fn per_minute_quota_silently_retries_same_model() {
    let body = rpc_error_body(vec![quota_failure("GenerateRequestsPerMinutePerProject")]);

    assert_eq!(
        classify_google_rpc_error(StatusCode::TOO_MANY_REQUESTS, &body),
        GeminiErrorDecision::RetrySameModel(None)
    );
}

#[test]
fn long_retry_delay_falls_back_model() {
    let body = rpc_error_body(vec![json!({
        "@type": "type.googleapis.com/google.rpc.RetryInfo",
        "retryDelay": "3600s"
    })]);

    assert_eq!(
        classify_google_rpc_error(StatusCode::TOO_MANY_REQUESTS, &body),
        GeminiErrorDecision::FallbackModel
    );
}

#[test]
fn per_day_quota_falls_back_model() {
    let body = rpc_error_body(vec![quota_failure("GenerateRequestsPerDayPerProject")]);

    assert_eq!(
        classify_google_rpc_error(StatusCode::TOO_MANY_REQUESTS, &body),
        GeminiErrorDecision::FallbackModel
    );
}

#[test]
fn token_context_errors_do_not_retry() {
    let body = json!({
        "error": {
            "status": "INVALID_ARGUMENT",
            "message": "The input token count exceeds the context window."
        }
    })
    .to_string();

    assert_eq!(
        classify_google_rpc_error(StatusCode::BAD_REQUEST, &body),
        GeminiErrorDecision::ContextWindowExceeded
    );
}

#[test]
fn malformed_rpc_body_does_not_retry() {
    assert_eq!(
        classify_google_rpc_error(StatusCode::TOO_MANY_REQUESTS, "{not json"),
        GeminiErrorDecision::NoRetry
    );
}

fn rpc_error_body(details: Vec<serde_json::Value>) -> String {
    json!({
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "message": "Quota exceeded.",
            "details": details
        }
    })
    .to_string()
}

fn quota_failure(quota_id: &str) -> serde_json::Value {
    json!({
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [{
            "quotaId": quota_id
        }]
    })
}
