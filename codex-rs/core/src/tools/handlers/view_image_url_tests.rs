use std::io::Cursor;
use std::sync::Arc;

use codex_protocol::models::DEFAULT_IMAGE_DETAIL;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ResponseInputItem;
use image::DynamicImage;
use image::ImageBuffer;
use image::ImageFormat;
use image::Rgb;
use image::Rgba;
use pretty_assertions::assert_eq;
use reqwest::Client;
use serde_json::json;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use tokio::sync::oneshot;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

use super::*;
use crate::session::tests::make_session_and_context;
use crate::tools::context::ToolCallSource;
use crate::turn_diff_tracker::TurnDiffTracker;

fn image_bytes(format: ImageFormat) -> Vec<u8> {
    let mut encoded = Cursor::new(Vec::new());
    if format == ImageFormat::Jpeg {
        let image = ImageBuffer::from_pixel(2, 2, Rgb([10u8, 20, 30]));
        DynamicImage::ImageRgb8(image)
            .write_to(&mut encoded, format)
            .expect("encode image");
    } else {
        let image = ImageBuffer::from_pixel(2, 2, Rgba([10u8, 20, 30, 255]));
        DynamicImage::ImageRgba8(image)
            .write_to(&mut encoded, format)
            .expect("encode image");
    }
    encoded.into_inner()
}

async fn invocation(arguments: serde_json::Value) -> ToolInvocation {
    let (session, turn) = make_session_and_context().await;
    invocation_with_turn(arguments, session, turn)
}

fn invocation_with_turn(
    arguments: serde_json::Value,
    session: crate::session::session::Session,
    turn: crate::session::turn_context::TurnContext,
) -> ToolInvocation {
    ToolInvocation {
        session: Arc::new(session),
        turn: Arc::new(turn),
        cancellation_token: tokio_util::sync::CancellationToken::new(),
        tracker: Arc::new(Mutex::new(TurnDiffTracker::new())),
        call_id: "call-view-image-url".to_string(),
        tool_name: ToolName::plain(VIEW_IMAGE_URL_TOOL_NAME),
        source: ToolCallSource::Direct,
        payload: ToolPayload::Function {
            arguments: arguments.to_string(),
        },
    }
}

#[test]
fn spec_exposes_required_url() {
    let ToolSpec::Function(tool) = create_view_image_url_tool() else {
        panic!("expected function tool");
    };
    let parameters = serde_json::to_value(&tool.parameters).expect("serialize parameters");

    assert_eq!(tool.name, VIEW_IMAGE_URL_TOOL_NAME);
    assert_eq!(parameters["required"], json!(["url"]));
    assert!(parameters["properties"].get("url").is_some());
}

#[tokio::test]
async fn rejects_unsupported_image_modality() {
    let (session, mut turn) = make_session_and_context().await;
    turn.model_info.input_modalities.clear();
    let invocation = invocation_with_turn(
        json!({"url": "https://example.com/chart.png"}),
        session,
        turn,
    );

    let result = ViewImageUrlHandler::new(Client::new())
        .handle(invocation)
        .await;

    let Err(FunctionCallError::RespondToModel(message)) = result else {
        panic!("expected unsupported modality error");
    };
    assert_eq!(message, VIEW_IMAGE_URL_UNSUPPORTED_MESSAGE);
}

#[tokio::test]
async fn rejects_invalid_and_non_http_urls() {
    for (url, expected) in [
        ("not a url", "url must be a valid URL:"),
        (
            "file:///tmp/chart.png",
            "unsupported URL scheme `file`; use http or https",
        ),
    ] {
        let result = ViewImageUrlHandler::new(Client::new())
            .handle(invocation(json!({ "url": url })).await)
            .await;

        let Err(FunctionCallError::RespondToModel(message)) = result else {
            panic!("expected URL validation error");
        };
        assert!(
            message.starts_with(expected),
            "expected {expected:?}, got {message:?}"
        );
    }
}

#[tokio::test]
async fn rejects_missing_url_argument() {
    let result = ViewImageUrlHandler::new(Client::new())
        .handle(invocation(json!({})).await)
        .await;

    let Err(FunctionCallError::RespondToModel(message)) = result else {
        panic!("expected argument parsing error");
    };
    assert!(
        message.starts_with("failed to parse function arguments:"),
        "unexpected message: {message}"
    );
}

#[tokio::test]
async fn rejects_non_image_content_type() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/page"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("hello", "text/plain"))
        .mount(&server)
        .await;
    let url = format!("{}/page", server.uri());

    let result = ViewImageUrlHandler::new(Client::new())
        .handle(invocation(json!({ "url": url })).await)
        .await;

    let Err(FunctionCallError::RespondToModel(message)) = result else {
        panic!("expected content-type error");
    };
    assert!(
        message.contains("expected an image response"),
        "unexpected message: {message}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn rejects_oversized_content_length() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener should bind");
    let addr = listener
        .local_addr()
        .expect("test listener should have local address");
    let (done_tx, done_rx) = oneshot::channel();
    let accept_task = tokio::spawn(async move {
        let (mut socket, _) = listener
            .accept()
            .await
            .expect("test listener should accept one connection");
        let mut request = vec![0u8; 1024];
        let _ = socket
            .read(&mut request)
            .await
            .expect("read request headers");
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\n\r\n",
            MAX_REMOTE_IMAGE_BYTES + 1
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write response headers");
        socket.write_all(&[0]).await.expect("write first body byte");
        let _ = done_rx.await;
    });
    let url = format!("http://{addr}/image.png");

    let result = ViewImageUrlHandler::new(Client::new())
        .handle(invocation(json!({ "url": url })).await)
        .await;
    let _ = done_tx.send(());
    accept_task
        .await
        .expect("test listener task should complete");

    let Err(FunctionCallError::RespondToModel(message)) = result else {
        panic!("expected oversized content-length error");
    };
    assert!(
        message.contains("exceeds the 10485760 byte limit"),
        "unexpected message: {message}"
    );
}

#[tokio::test]
async fn rejects_stream_that_exceeds_byte_cap() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener should bind");
    let addr = listener
        .local_addr()
        .expect("test listener should have local address");
    let accept_task = tokio::spawn(async move {
        let (mut socket, _) = listener
            .accept()
            .await
            .expect("test listener should accept one connection");
        let mut request = vec![0u8; 1024];
        let _ = socket
            .read(&mut request)
            .await
            .expect("read request headers");
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nConnection: close\r\n\r\n")
            .await
            .expect("write response headers");
        let chunk = vec![0u8; 8192];
        for _ in 0..=((MAX_REMOTE_IMAGE_BYTES / chunk.len()) + 1) {
            if socket.write_all(&chunk).await.is_err() {
                return;
            }
        }
    });
    let url = format!("http://{addr}/image.png");

    let result = ViewImageUrlHandler::new(Client::new())
        .handle(invocation(json!({ "url": url })).await)
        .await;
    accept_task
        .await
        .expect("test listener task should complete");

    let Err(FunctionCallError::RespondToModel(message)) = result else {
        panic!("expected streamed byte cap error");
    };
    assert!(
        message.contains("response exceeds the 10485760 byte limit"),
        "unexpected message: {message}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn fetches_remote_image_as_input_image_content_item() {
    for (extension, format, mime) in [
        ("png", ImageFormat::Png, "image/png"),
        ("jpg", ImageFormat::Jpeg, "image/jpeg"),
        ("webp", ImageFormat::WebP, "image/webp"),
    ] {
        let server = MockServer::start().await;
        let route = format!("/image.{extension}");
        Mock::given(method("GET"))
            .and(path(route.as_str()))
            .respond_with(ResponseTemplate::new(200).set_body_raw(image_bytes(format), mime))
            .mount(&server)
            .await;
        let url = format!("{}{route}", server.uri());
        let payload = ToolPayload::Function {
            arguments: json!({ "url": url }).to_string(),
        };

        let output = ViewImageUrlHandler::new(Client::new())
            .handle(invocation(json!({ "url": url })).await)
            .await
            .expect("view_image_url should succeed");
        let item = output.to_response_item("call-view-image-url", &payload);

        let ResponseInputItem::FunctionCallOutput { output, .. } = item else {
            panic!("expected function call output");
        };
        let success = output.success;
        let FunctionCallOutputBody::ContentItems(items) = output.body else {
            panic!("expected content items");
        };
        assert_eq!(items.len(), 1);
        let FunctionCallOutputContentItem::InputImage { image_url, detail } = &items[0] else {
            panic!("expected input image");
        };
        assert!(
            image_url.starts_with(&format!("data:{mime};base64,")),
            "unexpected image URL: {image_url}"
        );
        assert_eq!(*detail, Some(DEFAULT_IMAGE_DETAIL));
        assert_eq!(success, Some(true));
    }
}
