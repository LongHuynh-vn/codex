use std::sync::LazyLock;

pub const PLAN: &str = include_str!("../templates/plan.md");
pub const PLAN_GEMINI_ADDENDUM: &str = include_str!("../templates/plan_gemini_addendum.md");
pub const DEFAULT: &str = include_str!("../templates/default.md");
pub const EXECUTE: &str = include_str!("../templates/execute.md");
pub const PAIR_PROGRAMMING: &str = include_str!("../templates/pair_programming.md");

static PLAN_FOR_GEMINI: LazyLock<String> =
    LazyLock::new(|| format!("{PLAN}\n{PLAN_GEMINI_ADDENDUM}"));

pub fn plan_for_gemini() -> &'static str {
    &PLAN_FOR_GEMINI
}
