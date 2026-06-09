use std::path::PathBuf;

use codex_protocol::models::DEFAULT_IMAGE_DETAIL;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::openai_models::InputModality;
use codex_tools::ToolName;
use codex_tools::ToolSpec;
use codex_utils_image::PromptImageMode;
use codex_utils_image::load_for_prompt_bytes;
use futures::StreamExt;
use http::header::CONTENT_LENGTH;
use http::header::CONTENT_TYPE;
use reqwest::Client;
use serde::Deserialize;
use url::Url;

use crate::function_tool::FunctionCallError;
use crate::tools::context::FunctionToolOutput;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::context::boxed_tool_output;
use crate::tools::handlers::parse_arguments;
use crate::tools::handlers::view_image_url_spec::VIEW_IMAGE_URL_TOOL_NAME;
use crate::tools::handlers::view_image_url_spec::create_view_image_url_tool;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::ToolExecutor;

const VIEW_IMAGE_URL_UNSUPPORTED_MESSAGE: &str =
    "view_image_url is not allowed because you do not support image inputs";
const MAX_REMOTE_IMAGE_BYTES: usize = 10 * 1024 * 1024;

pub(crate) struct ViewImageUrlHandler {
    client: Client,
}

impl ViewImageUrlHandler {
    pub(crate) fn new(client: Client) -> Self {
        Self { client }
    }
}

#[derive(Deserialize)]
struct ViewImageUrlArgs {
    url: String,
}

#[async_trait::async_trait]
impl ToolExecutor<ToolInvocation> for ViewImageUrlHandler {
    fn tool_name(&self) -> ToolName {
        ToolName::plain(VIEW_IMAGE_URL_TOOL_NAME)
    }

    fn spec(&self) -> ToolSpec {
        create_view_image_url_tool()
    }

    fn supports_parallel_tool_calls(&self) -> bool {
        true
    }

    async fn handle(
        &self,
        invocation: ToolInvocation,
    ) -> Result<Box<dyn crate::tools::context::ToolOutput>, FunctionCallError> {
        if !invocation
            .turn
            .model_info
            .input_modalities
            .contains(&InputModality::Image)
        {
            return Err(FunctionCallError::RespondToModel(
                VIEW_IMAGE_URL_UNSUPPORTED_MESSAGE.to_string(),
            ));
        }

        let ToolPayload::Function { arguments } = invocation.payload else {
            return Err(FunctionCallError::RespondToModel(
                "view_image_url handler received unsupported payload".to_string(),
            ));
        };
        let args: ViewImageUrlArgs = parse_arguments(&arguments)?;
        let url = Url::parse(args.url.trim()).map_err(|err| {
            FunctionCallError::RespondToModel(format!("url must be a valid URL: {err}"))
        })?;
        match url.scheme() {
            "http" | "https" => {}
            scheme => {
                return Err(FunctionCallError::RespondToModel(format!(
                    "unsupported URL scheme `{scheme}`; use http or https"
                )));
            }
        }

        let response = self.client.get(url.clone()).send().await.map_err(|err| {
            FunctionCallError::RespondToModel(format!("view_image_url request failed: {err}"))
        })?;
        let status = response.status();
        if !status.is_success() {
            return Err(FunctionCallError::RespondToModel(format!(
                "view_image_url request failed for {url} with status {status}"
            )));
        }

        let headers = response.headers();
        let content_type = headers
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("unknown")
            .to_string();
        if !is_supported_image_content_type(&content_type) {
            return Err(FunctionCallError::RespondToModel(format!(
                "view_image_url expected an image response, got Content-Type `{content_type}`"
            )));
        }
        if let Some(content_length) = headers
            .get(CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<usize>().ok())
            && content_length > MAX_REMOTE_IMAGE_BYTES
        {
            return Err(FunctionCallError::RespondToModel(format!(
                "view_image_url image is too large: {content_length} bytes exceeds the {MAX_REMOTE_IMAGE_BYTES} byte limit"
            )));
        }

        let bytes = read_response_bytes_limited(response, MAX_REMOTE_IMAGE_BYTES).await?;
        let path_hint = path_hint_for_url(&url);
        let image = load_for_prompt_bytes(path_hint.as_path(), bytes, PromptImageMode::ResizeToFit)
            .map_err(|error| {
                FunctionCallError::RespondToModel(format!(
                    "unable to process image from `{url}`: {error}"
                ))
            })?;
        let image_url = image.into_data_url();

        Ok(boxed_tool_output(FunctionToolOutput::from_content(
            vec![FunctionCallOutputContentItem::InputImage {
                image_url,
                detail: Some(DEFAULT_IMAGE_DETAIL),
            }],
            /*success*/ Some(true),
        )))
    }
}

impl CoreToolRuntime for ViewImageUrlHandler {}

fn is_supported_image_content_type(content_type: &str) -> bool {
    let media_type = content_type
        .split(';')
        .next()
        .map(str::trim)
        .unwrap_or_default()
        .to_ascii_lowercase();
    matches!(
        media_type.as_str(),
        "image/png" | "image/jpeg" | "image/gif" | "image/webp"
    )
}

async fn read_response_bytes_limited(
    response: reqwest::Response,
    max_bytes: usize,
) -> Result<Vec<u8>, FunctionCallError> {
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| {
            FunctionCallError::RespondToModel(format!("view_image_url response read failed: {err}"))
        })?;
        let new_len = body.len().checked_add(chunk.len()).ok_or_else(|| {
            FunctionCallError::RespondToModel(
                "view_image_url image is too large to read".to_string(),
            )
        })?;
        if new_len > max_bytes {
            return Err(FunctionCallError::RespondToModel(format!(
                "view_image_url image is too large: response exceeds the {max_bytes} byte limit"
            )));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn path_hint_for_url(url: &Url) -> PathBuf {
    let file_name = url
        .path_segments()
        .and_then(|mut segments| segments.rfind(|segment| !segment.is_empty()))
        .unwrap_or("remote-image");
    PathBuf::from(file_name)
}

#[cfg(test)]
#[path = "view_image_url_tests.rs"]
mod tests;
