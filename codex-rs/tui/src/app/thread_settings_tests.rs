use codex_app_server_protocol::ThreadSettingsUpdateParams;
use codex_protocol::protocol::GeminiSearchMode;
use pretty_assertions::assert_eq;

#[test]
fn thread_settings_update_has_changes_covers_gemini_search_mode() {
    let unchanged = ThreadSettingsUpdateParams {
        thread_id: "thread_123".to_string(),
        ..Default::default()
    };
    assert_eq!(super::thread_settings_update_has_changes(&unchanged), false);

    let changed = ThreadSettingsUpdateParams {
        thread_id: "thread_123".to_string(),
        gemini_search_mode: Some(GeminiSearchMode::Hybrid),
        ..Default::default()
    };
    assert_eq!(super::thread_settings_update_has_changes(&changed), true);
}
