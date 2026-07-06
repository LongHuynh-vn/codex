//! Centralized motion primitives for the TUI.
//!
//! Callers choose an explicit reduced-motion fallback here instead of reaching
//! directly for time-varying spinner or shimmer helpers.

use std::sync::OnceLock;
use std::time::Duration;

use ratatui::style::Color;
use ratatui::style::Stylize;
use ratatui::text::Span;

use crate::color::blend;
use crate::shimmer::shimmer_spans;
use crate::terminal_palette::default_fg;

const BLOOM_FRAME_INTERVAL_MS: u128 = 110;
const STALL_GRACE: Duration = Duration::from_secs(3);
const STALL_RAMP: Duration = Duration::from_secs(2);
pub(crate) const STALL_TINT_RGB: (u8, u8, u8) = (171, 43, 63);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MotionMode {
    Animated,
    Reduced,
}

impl MotionMode {
    pub(crate) fn from_animations_enabled(animations_enabled: bool) -> Self {
        if animations_enabled {
            Self::Animated
        } else {
            Self::Reduced
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReducedMotionIndicator {
    Hidden,
    StaticBullet,
}

pub(crate) fn activity_indicator(
    elapsed: Duration,
    stall_intensity: f32,
    motion_mode: MotionMode,
    reduced_motion_indicator: ReducedMotionIndicator,
) -> Option<Span<'static>> {
    match motion_mode {
        MotionMode::Animated => Some(animated_activity_indicator(elapsed, stall_intensity)),
        MotionMode::Reduced => match reduced_motion_indicator {
            ReducedMotionIndicator::Hidden => None,
            ReducedMotionIndicator::StaticBullet => Some("•".dim()),
        },
    }
}

pub(crate) fn shimmer_text(text: &str, motion_mode: MotionMode) -> Vec<Span<'static>> {
    match motion_mode {
        MotionMode::Animated => shimmer_spans(text),
        MotionMode::Reduced => {
            if text.is_empty() {
                Vec::new()
            } else {
                vec![text.to_string().into()]
            }
        }
    }
}

pub(crate) fn stall_intensity(since_activity: Duration, motion_mode: MotionMode) -> f32 {
    if since_activity <= STALL_GRACE {
        return 0.0;
    }

    match motion_mode {
        MotionMode::Animated => {
            let ramp_elapsed = since_activity.saturating_sub(STALL_GRACE);
            (ramp_elapsed.as_secs_f32() / STALL_RAMP.as_secs_f32()).clamp(0.0, 1.0)
        }
        MotionMode::Reduced => 1.0,
    }
}

fn bloom_frames(is_macos: bool, is_ghostty_term: bool) -> [&'static str; 12] {
    let eight_spoked = if is_macos { "✳" } else { "*" };
    let teardrop = if is_ghostty_term { "*" } else { "✽" };
    let base = ["·", "✢", eight_spoked, "✶", "✻", teardrop];
    [
        base[0], base[1], base[2], base[3], base[4], base[5], base[5], base[4], base[3], base[2],
        base[1], base[0],
    ]
}

fn runtime_bloom_frames() -> &'static [&'static str; 12] {
    static FRAMES: OnceLock<[&'static str; 12]> = OnceLock::new();
    FRAMES.get_or_init(|| {
        let is_ghostty = std::env::var("TERM").is_ok_and(|term| term == "xterm-ghostty");
        bloom_frames(cfg!(target_os = "macos"), is_ghostty)
    })
}

fn bloom_frame_glyph(frames: &[&'static str; 12], elapsed: Duration) -> &'static str {
    let index = (elapsed.as_millis() / BLOOM_FRAME_INTERVAL_MS) as usize % frames.len();
    frames[index]
}

fn animated_activity_indicator(elapsed: Duration, stall_intensity: f32) -> Span<'static> {
    let glyph = bloom_frame_glyph(runtime_bloom_frames(), elapsed);
    if supports_color::on_cached(supports_color::Stream::Stdout)
        .map(|level| level.has_16m)
        .unwrap_or(false)
    {
        let mut span = shimmer_spans(glyph)
            .into_iter()
            .next()
            .unwrap_or_else(|| glyph.into());
        if stall_intensity > 0.0 {
            let base = match span.style.fg {
                Some(Color::Rgb(r, g, b)) => (r, g, b),
                _ => default_fg().unwrap_or((128, 128, 128)),
            };
            let (r, g, b) = blend(STALL_TINT_RGB, base, stall_intensity.clamp(0.0, 1.0));
            #[allow(clippy::disallowed_methods)]
            {
                span.style = span.style.fg(Color::Rgb(r, g, b));
            }
        }
        span
    } else if stall_intensity > 0.5 {
        glyph.red()
    } else {
        glyph.into()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::path::PathBuf;
    use std::time::Duration;

    use pretty_assertions::assert_eq;
    use unicode_width::UnicodeWidthStr;

    use super::*;

    #[test]
    fn reduced_motion_activity_indicator_uses_explicit_fallback() {
        assert_eq!(
            activity_indicator(
                Duration::ZERO,
                /*stall_intensity*/ 0.0,
                MotionMode::Reduced,
                ReducedMotionIndicator::Hidden,
            ),
            None
        );
        assert_eq!(
            activity_indicator(
                Duration::ZERO,
                /*stall_intensity*/ 0.0,
                MotionMode::Reduced,
                ReducedMotionIndicator::StaticBullet,
            ),
            Some("•".dim())
        );
    }

    #[test]
    fn bloom_frames_ping_pong_and_platform_substitutions() {
        assert_eq!(
            bloom_frames(/*is_macos*/ true, /*is_ghostty_term*/ false),
            ["·", "✢", "✳", "✶", "✻", "✽", "✽", "✻", "✶", "✳", "✢", "·"]
        );
        assert_eq!(
            bloom_frames(/*is_macos*/ false, /*is_ghostty_term*/ false)[2],
            "*"
        );
        assert_eq!(
            bloom_frames(/*is_macos*/ true, /*is_ghostty_term*/ true)[5],
            "*"
        );
    }

    #[test]
    fn bloom_frame_glyph_advances_on_interval() {
        let frames = bloom_frames(/*is_macos*/ true, /*is_ghostty_term*/ false);
        assert_eq!(bloom_frame_glyph(&frames, Duration::ZERO), "·");
        assert_eq!(bloom_frame_glyph(&frames, Duration::from_millis(109)), "·");
        assert_eq!(bloom_frame_glyph(&frames, Duration::from_millis(110)), "✢");
        assert_eq!(bloom_frame_glyph(&frames, Duration::from_millis(550)), "✽");
        assert_eq!(bloom_frame_glyph(&frames, Duration::from_millis(660)), "✽");
        assert_eq!(bloom_frame_glyph(&frames, Duration::from_millis(1210)), "·");
        assert_eq!(bloom_frame_glyph(&frames, Duration::from_millis(1320)), "·");
    }

    #[test]
    fn bloom_frames_are_single_column() {
        for is_macos in [false, true] {
            for is_ghostty_term in [false, true] {
                for frame in bloom_frames(is_macos, is_ghostty_term) {
                    assert_eq!(UnicodeWidthStr::width(frame), 1);
                }
            }
        }
    }

    #[test]
    fn stall_intensity_respects_motion_mode() {
        assert_eq!(
            stall_intensity(Duration::from_secs(3), MotionMode::Animated),
            0.0
        );
        assert_eq!(
            stall_intensity(Duration::from_secs(4), MotionMode::Animated),
            0.5
        );
        assert_eq!(
            stall_intensity(Duration::from_secs(5), MotionMode::Animated),
            1.0
        );
        assert_eq!(
            stall_intensity(Duration::from_secs(10), MotionMode::Animated),
            1.0
        );
        assert_eq!(
            stall_intensity(Duration::from_millis(3001), MotionMode::Reduced),
            1.0
        );
    }

    #[test]
    fn reduced_motion_shimmer_text_is_plain_text() {
        assert_eq!(
            shimmer_text("Loading", MotionMode::Reduced),
            vec!["Loading".into()]
        );
        assert_eq!(
            shimmer_text("", MotionMode::Reduced),
            Vec::<Span<'static>>::new()
        );
    }

    #[test]
    fn animation_primitives_are_only_used_by_motion_module() {
        let direct_spinner = regex_lite::Regex::new(r"(^|[^A-Za-z0-9_])spinner\s*\(").unwrap();
        let direct_shimmer =
            regex_lite::Regex::new(r"(^|[^A-Za-z0-9_])shimmer_spans\s*\(").unwrap();
        let lib_rs = codex_utils_cargo_bin::find_resource!("src/lib.rs")
            .expect("failed to locate TUI source");
        let src_dir = lib_rs.parent().expect("lib.rs should have a parent");

        let mut source_files = Vec::new();
        collect_rust_files(src_dir, &mut source_files).expect("failed to collect TUI source files");

        let mut violations = Vec::new();
        for path in source_files {
            let relative_path = path
                .strip_prefix(src_dir)
                .expect("source file should be under src")
                .to_string_lossy()
                .replace('\\', "/");
            if animation_primitive_allowlisted_path(&relative_path) {
                continue;
            }

            let contents = fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("failed to read {relative_path}: {err}"));
            for (line_number, line) in contents.lines().enumerate() {
                let code = line.split_once("//").map_or(line, |(code, _)| code);
                if direct_spinner.is_match(code) {
                    violations.push(format!(
                        "{relative_path}:{} contains a direct `spinner(...)` call; use crate::motion instead",
                        line_number + 1
                    ));
                }
                if direct_shimmer.is_match(code) {
                    violations.push(format!(
                        "{relative_path}:{} contains a direct `shimmer_spans(...)` call; use crate::motion instead",
                        line_number + 1
                    ));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "direct animation primitive usage found:\n{}",
            violations.join("\n")
        );
    }

    fn collect_rust_files(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                collect_rust_files(&path, files)?;
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                files.push(path);
            }
        }
        Ok(())
    }

    fn animation_primitive_allowlisted_path(relative_path: &str) -> bool {
        matches!(relative_path, "motion.rs" | "shimmer.rs")
    }
}
