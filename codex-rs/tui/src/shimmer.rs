use std::sync::OnceLock;
use std::time::Duration;
use std::time::Instant;

use ratatui::style::Color;
use ratatui::style::Modifier;
use ratatui::style::Style;
use ratatui::text::Span;

use crate::color::blend;
use crate::terminal_palette::default_bg;
use crate::terminal_palette::default_fg;

static PROCESS_START: OnceLock<Instant> = OnceLock::new();

fn elapsed_since_start() -> Duration {
    let start = PROCESS_START.get_or_init(Instant::now);
    start.elapsed()
}

/// Base coloring for shimmer text on true-color terminals.
///
/// `Default` keeps the terminal foreground; `Gradient` interpolates the given
/// anchors across the text as a static per-char base. Non-true-color terminals
/// ignore the palette entirely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShimmerPalette {
    Default,
    Gradient(&'static [(u8, u8, u8)]),
}

/// Gemini brand gradient for the status spinner verb: blue → purple → pink.
pub(crate) const GEMINI_VERB_PALETTE: ShimmerPalette =
    ShimmerPalette::Gradient(&[(0x47, 0x96, 0xE3), (0x91, 0x77, 0xC7), (0xD6, 0x64, 0x7E)]);
/// Muted teal accent for running-hook headers.
pub(crate) const HOOK_ACCENT_PALETTE: ShimmerPalette =
    ShimmerPalette::Gradient(&[(0x4E, 0x9A, 0x8E), (0x7F, 0xB8, 0xA8)]);
/// Muted violet accent for plugin loading text.
pub(crate) const PLUGIN_ACCENT_PALETTE: ShimmerPalette =
    ShimmerPalette::Gradient(&[(0x8E, 0x7C, 0xC3), (0xB3, 0x9D, 0xDB)]);

/// Piecewise-linear sample of gradient `anchors` at position `t` in [0, 1].
fn gradient_color_at(anchors: &[(u8, u8, u8)], t: f32) -> (u8, u8, u8) {
    match anchors {
        [] => (128, 128, 128),
        [only] => *only,
        _ => {
            let segments = (anchors.len() - 1) as f32;
            let scaled = t.clamp(0.0, 1.0) * segments;
            let index = (scaled as usize).min(anchors.len() - 2);
            let within = scaled - index as f32;
            blend(anchors[index + 1], anchors[index], within)
        }
    }
}

pub(crate) fn shimmer_spans(text: &str, palette: ShimmerPalette) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }
    // Use time-based sweep synchronized to process start.
    let padding = 10usize;
    let period = chars.len() + padding * 2;
    let sweep_seconds = 2.0f32;
    let pos_f =
        (elapsed_since_start().as_secs_f32() % sweep_seconds) / sweep_seconds * (period as f32);
    let pos = pos_f as usize;
    let has_true_color = supports_color::on_cached(supports_color::Stream::Stdout)
        .map(|level| level.has_16m)
        .unwrap_or(false);
    let band_half_width = 5.0;

    let mut spans: Vec<Span<'static>> = Vec::with_capacity(chars.len());
    let default_base = default_fg().unwrap_or((128, 128, 128));
    let highlight_color = default_bg().unwrap_or((255, 255, 255));
    for (i, ch) in chars.iter().enumerate() {
        let i_pos = i as isize + padding as isize;
        let pos = pos as isize;
        let dist = (i_pos - pos).abs() as f32;

        let t = if dist <= band_half_width {
            let x = std::f32::consts::PI * (dist / band_half_width);
            0.5 * (1.0 + x.cos())
        } else {
            0.0
        };
        let style = if has_true_color {
            let base_color = match palette {
                ShimmerPalette::Default => default_base,
                ShimmerPalette::Gradient(anchors) => gradient_color_at(
                    anchors,
                    i as f32 / (chars.len().saturating_sub(1)).max(1) as f32,
                ),
            };
            let highlight = t.clamp(0.0, 1.0);
            let (r, g, b) = blend(highlight_color, base_color, highlight * 0.9);
            // Allow custom RGB colors, as the implementation is thoughtfully
            // adjusting the level of the default foreground color.
            #[allow(clippy::disallowed_methods)]
            {
                Style::default()
                    .fg(Color::Rgb(r, g, b))
                    .add_modifier(Modifier::BOLD)
            }
        } else {
            color_for_level(t)
        };
        spans.push(Span::styled(ch.to_string(), style));
    }
    spans
}

fn color_for_level(intensity: f32) -> Style {
    // Tune fallback styling so the shimmer band reads even without RGB support.
    if intensity < 0.2 {
        Style::default().add_modifier(Modifier::DIM)
    } else if intensity < 0.6 {
        Style::default()
    } else {
        Style::default().add_modifier(Modifier::BOLD)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn gradient_color_at_interpolates_anchors() {
        let ShimmerPalette::Gradient(anchors) = GEMINI_VERB_PALETTE else {
            panic!("verb palette should be a gradient");
        };
        assert_eq!(gradient_color_at(anchors, 0.0), (0x47, 0x96, 0xE3));
        assert_eq!(gradient_color_at(anchors, 0.5), (0x91, 0x77, 0xC7));
        assert_eq!(gradient_color_at(anchors, 1.0), (0xD6, 0x64, 0x7E));

        let two_anchors = &[(0, 0, 0), (200, 100, 50)];
        assert_eq!(gradient_color_at(two_anchors, 0.0), (0, 0, 0));
        assert_eq!(gradient_color_at(two_anchors, 0.5), (100, 50, 25));
        assert_eq!(gradient_color_at(two_anchors, 1.0), (200, 100, 50));
    }
}
