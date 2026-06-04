use pretty_assertions::assert_eq;
use serde_json::json;

use super::sanitize_tool_parameters;

#[test]
fn strips_gemini_unsupported_schema_fields_without_flattening_shape() {
    let sanitized = sanitize_tool_parameters(json!({
        "type": "object",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Known bad schema",
        "additionalProperties": false,
        "$defs": {
            "Coordinates": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"}
                }
            }
        },
        "properties": {
            "city": {
                "type": "string",
                "format": "uri-reference",
                "default": "https://example.com"
            },
            "metadata": {
                "$ref": "#/$defs/Coordinates",
                "type": "object",
                "examples": [{"source": "test"}],
                "additionalProperties": {
                    "type": "string",
                    "format": "duration"
                },
                "properties": {
                    "source": {
                        "type": "string",
                        "patternProperties": {"^x-": {"type": "string"}},
                        "readOnly": true
                    },
                    "nested": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "integer"}
                        }
                    }
                }
            }
        },
        "required": ["city"]
    }));

    assert_eq!(
        sanitized,
        json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string"
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string"
                        },
                        "nested": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "integer"}
                            }
                        }
                    }
                }
            },
            "required": ["city"]
        })
    );
}
