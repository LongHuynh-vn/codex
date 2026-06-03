use codex_protocol::error::CodexErr;
use codex_protocol::error::ConnectionFailedError;
use codex_protocol::error::ResponseStreamFailed;
use codex_protocol::error::Result;
use codex_protocol::error::UnexpectedResponseError;
use reqwest::Response;

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

pub(crate) async fn ensure_success(response: Response, url: &str) -> Result<Response> {
    if response.status().is_success() {
        return Ok(response);
    }

    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    Err(CodexErr::UnexpectedStatus(UnexpectedResponseError {
        status,
        body,
        url: Some(url.to_string()),
        cf_ray: None,
        request_id: None,
        identity_authorization_error: None,
        identity_error_code: None,
    }))
}
