use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;

pub(crate) const VIEW_IMAGE_URL_TOOL_NAME: &str = "view_image_url";

pub(crate) fn create_view_image_url_tool() -> ToolSpec {
    ToolSpec::Function(ResponsesApiTool {
        name: VIEW_IMAGE_URL_TOOL_NAME.to_string(),
        description:
            "Fetch and view a remote HTTP or HTTPS image when visual inspection is needed."
                .to_string(),
        strict: false,
        defer_loading: None,
        parameters: JsonSchema::object(
            BTreeMap::from([(
                "url".to_string(),
                JsonSchema::string(Some("HTTP or HTTPS URL of the image to view.".to_string())),
            )]),
            Some(vec!["url".to_string()]),
            /*additional_properties*/ Some(false.into()),
        ),
        output_schema: Some(view_image_url_output_schema()),
    })
}

fn view_image_url_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": "Data URL for the loaded image."
            },
            "detail": {
                "type": "string",
                "enum": ["high"],
                "description": "Image detail hint returned by view_image_url."
            }
        },
        "required": ["image_url", "detail"],
        "additionalProperties": false
    })
}
