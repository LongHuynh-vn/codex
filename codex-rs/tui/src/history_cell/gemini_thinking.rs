//! Gemini thought-summary history cell.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use super::*;

#[derive(Debug)]
pub(crate) struct GeminiThinkingCell {
    content: String,
    cwd: PathBuf,
    visible: Arc<AtomicBool>,
}

impl GeminiThinkingCell {
    pub(crate) fn new(content: String, cwd: &Path, visible: Arc<AtomicBool>) -> Self {
        Self {
            content,
            cwd: cwd.to_path_buf(),
            visible,
        }
    }

    fn is_visible(&self) -> bool {
        self.visible.load(Ordering::Relaxed)
    }

    fn lines(&self, width: u16) -> Vec<Line<'static>> {
        if !self.is_visible() {
            return Vec::new();
        }

        let mut lines: Vec<Line<'static>> = Vec::new();
        append_markdown(
            &self.content,
            crate::width::usable_content_width_u16(width, /*reserved_cols*/ 2),
            Some(self.cwd.as_path()),
            &mut lines,
        );
        let summary_style = Style::default().dim().italic();
        let summary_lines = lines
            .into_iter()
            .map(|mut line| {
                line.spans = line
                    .spans
                    .into_iter()
                    .map(|span| span.patch_style(summary_style))
                    .collect();
                line
            })
            .collect::<Vec<_>>();

        adaptive_wrap_lines(
            &summary_lines,
            RtOptions::new(width as usize)
                .initial_indent("• ".dim().into())
                .subsequent_indent("  ".into()),
        )
    }
}

impl HistoryCell for GeminiThinkingCell {
    fn display_lines(&self, width: u16) -> Vec<Line<'static>> {
        self.lines(width)
    }

    fn transcript_lines(&self, width: u16) -> Vec<Line<'static>> {
        self.lines(width)
    }

    fn raw_lines(&self) -> Vec<Line<'static>> {
        if self.is_visible() {
            raw_lines_from_source(self.content.trim())
        } else {
            Vec::new()
        }
    }
}
