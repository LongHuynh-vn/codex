use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;
use codex_tools::ToolSpec;
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Tool {
    pub(crate) function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Debug, Serialize, PartialEq)]
pub(crate) struct FunctionDeclaration {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) parameters: Value,
}

pub(crate) fn build_tools(tools: &[ToolSpec]) -> Result<Option<Vec<Tool>>> {
    let mut declarations = Vec::new();
    for tool in tools {
        match tool {
            ToolSpec::Function(tool) => declarations.push(FunctionDeclaration {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: serde_json::to_value(&tool.parameters)?,
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
        Ok(Some(vec![Tool {
            function_declarations: declarations,
        }]))
    }
}
