use codex_api::ResponseEvent;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ReasoningItemReasoningSummary;
use codex_protocol::models::ResponseItem;
use pretty_assertions::assert_eq;

use super::*;
use crate::GeminiThoughtSummaryDisplay;

#[test]
fn accumulates_stream_until_finish_reason_and_captures_signature_usage() {
    let mut accumulator = StreamAccumulator::default();

    assert!(
        accumulator
            .process_event_data(
                r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Looking"}]}}]}"#
            )
            .unwrap()
            .is_empty()
    );
    assert!(!accumulator.is_finished());
    accumulator
        .process_event_data(
            r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"record_step","args":{"step":1}},"thoughtSignature":"sig-1"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2,"thoughtsTokenCount":7,"cachedContentTokenCount":3,"totalTokenCount":19}}"#,
        )
        .unwrap();
    assert!(accumulator.is_finished());

    let events = accumulator.finish().unwrap();
    assert!(matches!(
        events[0],
        ResponseEvent::OutputItemDone(ResponseItem::Message { .. })
    ));
    let ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
        name,
        arguments,
        thought_signature,
        ..
    }) = &events[1]
    else {
        panic!("expected function call event");
    };
    assert_eq!(name, "record_step");
    assert_eq!(arguments, r#"{"step":1}"#);
    assert_eq!(thought_signature.as_deref(), Some("sig-1"));

    let ResponseEvent::Completed {
        token_usage: Some(usage),
        ..
    } = &events[2]
    else {
        panic!("expected completed event with usage");
    };
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.cached_input_tokens, 3);
    assert_eq!(usage.output_tokens, 2);
    assert_eq!(usage.reasoning_output_tokens, 7);
    assert_eq!(usage.total_tokens, 19);
}

#[test]
fn parses_grounding_metadata_shape_without_changing_stream_output() {
    let mut accumulator = StreamAccumulator::default();

    accumulator
        .process_event_data(
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"grounded answer"}]},"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["query one","query two"],"searchEntryPoint":{"renderedContent":"<style>.x{}</style><div>search</div>"},"retrievalMetadata":{}}}]}"#,
        )
        .unwrap();

    let events = accumulator.finish().unwrap();
    assert_eq!(message_text_from_events(&events), "grounded answer");
}

#[test]
fn grounding_metadata_is_optional() {
    let mut accumulator = StreamAccumulator::default();

    accumulator
        .process_event_data(
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"plain answer"}]},"finishReason":"STOP"}]}"#,
        )
        .unwrap();

    let events = accumulator.finish().unwrap();
    assert_eq!(message_text_from_events(&events), "plain answer");
}

#[test]
fn ignores_unknown_grounding_metadata_fields() {
    let mut accumulator = StreamAccumulator::default();

    accumulator
        .process_event_data(
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"future answer"}]},"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["query"],"groundingChunks":[{"web":{"uri":"https://example.com"}}],"groundingSupports":[{"segment":{"startIndex":0,"endIndex":6}}],"futureField":{"nested":true}}}]}"#,
        )
        .unwrap();

    let events = accumulator.finish().unwrap();
    assert_eq!(message_text_from_events(&events), "future answer");
}

#[test]
fn retains_thought_text_in_reasoning_item_by_default() {
    let mut accumulator = StreamAccumulator::default();

    accumulator
        .process_event_data(
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"private reasoning","thought":true},{"text":"visible answer"}]},"finishReason":"STOP"}]}"#,
        )
        .unwrap();

    let events = accumulator.finish().unwrap();
    assert_eq!(events.len(), 3);
    let ResponseEvent::OutputItemDone(ResponseItem::Reasoning {
        summary,
        content,
        encrypted_content,
        ..
    }) = &events[0]
    else {
        panic!("expected reasoning item");
    };
    assert_eq!(
        summary,
        &vec![ReasoningItemReasoningSummary::SummaryText {
            text: "private reasoning".to_string()
        }]
    );
    assert_eq!(content, &None);
    assert_eq!(encrypted_content, &None);

    let ResponseEvent::OutputItemDone(ResponseItem::Message { content, .. }) = &events[1] else {
        panic!("expected visible assistant message");
    };
    assert_eq!(
        content,
        &vec![ContentItem::OutputText {
            text: "visible answer".to_string()
        }]
    );
}

#[test]
fn visible_thought_text_uses_reasoning_summary_channel() {
    let mut accumulator = StreamAccumulator::new(GeminiThoughtSummaryDisplay::Visible);

    accumulator
        .process_event_data(
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"private reasoning","thought":true},{"text":"visible answer"}]},"finishReason":"STOP"}]}"#,
        )
        .unwrap();

    let events = accumulator.finish().unwrap();
    assert_eq!(events.len(), 3);
    let ResponseEvent::OutputItemDone(ResponseItem::Reasoning {
        summary,
        content,
        encrypted_content,
        ..
    }) = &events[0]
    else {
        panic!("expected reasoning item");
    };
    assert_eq!(
        summary,
        &vec![ReasoningItemReasoningSummary::SummaryText {
            text: "private reasoning".to_string()
        }]
    );
    assert_eq!(content, &None);
    assert_eq!(encrypted_content, &None);

    let ResponseEvent::OutputItemDone(ResponseItem::Message { content, .. }) = &events[1] else {
        panic!("expected visible assistant message");
    };
    assert_eq!(
        content,
        &vec![ContentItem::OutputText {
            text: "visible answer".to_string()
        }]
    );
}

#[test]
fn thought_part_signature_after_unsigned_function_call_stays_on_call() {
    let mut accumulator = StreamAccumulator::default();

    accumulator
        .process_event_data(
            r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"record_step","args":{"step":1}}},{"text":"private reasoning","thought":true,"thoughtSignature":"sig-late"}]},"finishReason":"STOP"}]}"#,
        )
        .unwrap();

    let events = accumulator.finish().unwrap();
    assert_eq!(events.len(), 3);
    assert!(matches!(
        events[0],
        ResponseEvent::OutputItemDone(ResponseItem::Reasoning { .. })
    ));
    let ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
        name,
        thought_signature,
        ..
    }) = &events[1]
    else {
        panic!("expected function call");
    };
    assert_eq!(name, "record_step");
    assert_eq!(thought_signature.as_deref(), Some("sig-late"));
}

#[test]
fn standalone_thought_part_signature_is_preserved_with_summary() {
    let mut accumulator = StreamAccumulator::default();

    accumulator
        .process_event_data(
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"private reasoning","thought":true,"thoughtSignature":"sig-standalone"}]},"finishReason":"STOP"}]}"#,
        )
        .unwrap();

    let events = accumulator.finish().unwrap();
    assert_eq!(events.len(), 2);
    let ResponseEvent::OutputItemDone(ResponseItem::Reasoning {
        summary,
        content,
        encrypted_content,
        ..
    }) = &events[0]
    else {
        panic!("expected reasoning item for standalone signature");
    };
    assert_eq!(
        summary,
        &vec![ReasoningItemReasoningSummary::SummaryText {
            text: "private reasoning".to_string()
        }]
    );
    assert_eq!(content, &None);
    assert_eq!(encrypted_content.as_deref(), Some("sig-standalone"));
}

fn message_text_from_events(events: &[ResponseEvent]) -> &str {
    let ResponseEvent::OutputItemDone(ResponseItem::Message { content, .. }) = &events[0] else {
        panic!("expected message event");
    };
    match content.as_slice() {
        [ContentItem::OutputText { text }] => text.as_str(),
        _ => panic!("expected one output text item"),
    }
}
