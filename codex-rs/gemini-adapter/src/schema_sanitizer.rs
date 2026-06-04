use serde_json::Map;
use serde_json::Value;

const STRIPPED_SCHEMA_KEYS: &[&str] = &[
    "$defs",
    "$ref",
    "$schema",
    "additionalProperties",
    "default",
    "definitions",
    "deprecated",
    "examples",
    "format",
    "patternProperties",
    "readOnly",
    "title",
    "unevaluatedProperties",
    "writeOnly",
];

pub(crate) fn sanitize_tool_parameters(mut parameters: Value) -> Value {
    sanitize_schema_value(&mut parameters);
    parameters
}

fn sanitize_schema_value(value: &mut Value) {
    match value {
        Value::Object(map) => sanitize_schema_object(map),
        Value::Array(items) => {
            for item in items {
                sanitize_schema_value(item);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

fn sanitize_schema_object(map: &mut Map<String, Value>) {
    for key in STRIPPED_SCHEMA_KEYS {
        map.remove(*key);
    }

    for value in map.values_mut() {
        sanitize_schema_value(value);
    }
}

#[cfg(test)]
#[path = "schema_sanitizer_tests.rs"]
mod tests;
