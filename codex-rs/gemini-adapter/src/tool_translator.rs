use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;
use codex_tools::ToolSpec;
use serde::Serialize;
use serde_json::Value;

use crate::schema_sanitizer::sanitize_tool_parameters;

#[derive(Debug, Serialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum Tool {
    FunctionDeclarations {
        #[serde(rename = "functionDeclarations")]
        function_declarations: Vec<FunctionDeclaration>,
    },
    GoogleSearch {
        #[serde(rename = "googleSearch")]
        google_search: GoogleSearch,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GoogleSearchGrounding {
    Disabled,
    Enabled,
}

#[derive(Debug, Serialize, PartialEq)]
pub(crate) struct GoogleSearch {}

#[derive(Debug, Serialize, PartialEq)]
pub(crate) struct FunctionDeclaration {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) parameters: Value,
}

pub(crate) fn build_tools(
    tools: &[ToolSpec],
    google_search_grounding: GoogleSearchGrounding,
) -> Result<Option<Vec<Tool>>> {
    let mut declarations = Vec::new();
    for tool in tools {
        match tool {
            ToolSpec::Function(tool) => declarations.push(FunctionDeclaration {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: sanitize_tool_parameters(serde_json::to_value(&tool.parameters)?),
            }),
            ToolSpec::Namespace(_)
            | ToolSpec::ToolSearch { .. }
            | ToolSpec::ImageGeneration { .. }
            | ToolSpec::WebSearch { .. }
            | ToolSpec::Freeform(_) => {
                return Err(CodexErr::UnsupportedOperation(format!(
                    "Gemini Phase 1 only supports function tools; unsupported tool `{}`",
                    tool.name()
                )));
            }
        }
    }

    if declarations.is_empty() {
        Ok(None)
    } else {
        let mut tools = vec![Tool::FunctionDeclarations {
            function_declarations: declarations,
        }];
        if google_search_grounding == GoogleSearchGrounding::Enabled {
            tools.push(Tool::GoogleSearch {
                google_search: GoogleSearch {},
            });
        }
        Ok(Some(tools))
    }
}
