//! Patch summaries and image-tool transcript helpers.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use super::*;

const DIFF_COLLAPSE_THRESHOLD: usize = 20;

#[derive(Debug)]
pub(crate) struct PatchHistoryCell {
    changes: HashMap<PathBuf, FileChange>,
    cwd: PathBuf,
    diff_expanded: Arc<AtomicBool>,
}

impl HistoryCell for PatchHistoryCell {
    fn display_lines(&self, width: u16) -> Vec<Line<'static>> {
        let mut lines = create_diff_summary(&self.changes, &self.cwd, width as usize);
        let body = lines.len().saturating_sub(1);
        if !self.diff_expanded.load(Ordering::Relaxed) && body > DIFF_COLLAPSE_THRESHOLD {
            lines.truncate(1);
            lines.push(
                format!("  └ … {body} more lines — Option+X to expand")
                    .dim()
                    .into(),
            );
        }
        lines
    }

    fn raw_lines(&self) -> Vec<Line<'static>> {
        plain_lines(create_diff_summary(
            &self.changes,
            &self.cwd,
            RAW_DIFF_SUMMARY_WIDTH,
        ))
    }

    fn transcript_lines(&self, width: u16) -> Vec<Line<'static>> {
        create_diff_summary(&self.changes, &self.cwd, width as usize)
    }
}
/// Create a new `PendingPatch` cell that lists the file‑level summary of
/// a proposed patch. The summary lines should already be formatted (e.g.
/// "A path/to/file.rs").
pub(crate) fn new_patch_event(
    changes: HashMap<PathBuf, FileChange>,
    cwd: &Path,
    diff_expanded: Arc<AtomicBool>,
) -> PatchHistoryCell {
    PatchHistoryCell {
        changes,
        cwd: cwd.to_path_buf(),
        diff_expanded,
    }
}

pub(crate) fn new_patch_apply_failure(stderr: String) -> PlainHistoryCell {
    let mut lines: Vec<Line<'static>> = Vec::new();

    // Failure title
    lines.push(Line::from("✘ Failed to apply patch".magenta().bold()));

    if !stderr.trim().is_empty() {
        let output = output_lines(
            Some(&CommandOutput {
                exit_code: 1,
                formatted_output: String::new(),
                aggregated_output: stderr,
            }),
            OutputLinesParams {
                line_limit: TOOL_CALL_MAX_LINES,
                only_err: true,
                include_angle_pipe: true,
                include_prefix: true,
            },
        );
        lines.extend(output.lines);
    }

    PlainHistoryCell { lines }
}

pub(crate) fn new_view_image_tool_call(path: AbsolutePathBuf, cwd: &Path) -> PlainHistoryCell {
    let display_path = display_path_for(path.as_path(), cwd);

    let lines: Vec<Line<'static>> = vec![
        vec!["• ".dim(), "Viewed Image".bold()].into(),
        vec!["  └ ".dim(), display_path.dim()].into(),
    ];

    PlainHistoryCell { lines }
}

pub(crate) fn new_image_generation_call(
    call_id: String,
    revised_prompt: Option<String>,
    saved_path: Option<AbsolutePathBuf>,
) -> PlainHistoryCell {
    let detail = revised_prompt.unwrap_or_else(|| call_id.clone());

    let mut lines: Vec<Line<'static>> = vec![
        vec!["• ".dim(), "Generated Image:".bold()].into(),
        vec!["  └ ".dim(), detail.dim()].into(),
    ];
    if let Some(saved_path) = saved_path {
        let saved_path = Url::from_file_path(saved_path.as_path())
            .map(|url| url.to_string())
            .unwrap_or_else(|_| saved_path.display().to_string());
        lines.push(vec!["  └ ".dim(), "Saved to: ".dim(), saved_path.into()].into());
    }

    PlainHistoryCell { lines }
}
