//! Whimsical per-turn status verbs shown while the agent is working.

pub(crate) const SPINNER_VERBS: [&str; 54] = [
    "Aligning",
    "Arranging",
    "Assembling",
    "Baking",
    "Balancing",
    "Braiding",
    "Brewing",
    "Burnishing",
    "Calibrating",
    "Carving",
    "Charting",
    "Chiseling",
    "Churning",
    "Cogitating",
    "Combing",
    "Composing",
    "Deliberating",
    "Distilling",
    "Doodling",
    "Drafting",
    "Etching",
    "Filtering",
    "Folding",
    "Foraging",
    "Forging",
    "Gathering",
    "Incubating",
    "Inking",
    "Juggling",
    "Kneading",
    "Knitting",
    "Mapping",
    "Mulling",
    "Musing",
    "Noodling",
    "Orchestrating",
    "Percolating",
    "Perusing",
    "Polishing",
    "Pondering",
    "Refining",
    "Ruminating",
    "Sculpting",
    "Sifting",
    "Simmering",
    "Sketching",
    "Steeping",
    "Stitching",
    "Threading",
    "Tinkering",
    "Untangling",
    "Weaving",
    "Whisking",
    "Whittling",
];

/// Deterministic verb for rotation `phase` of a turn seeded with `seed`.
///
/// The stride is co-prime with the verb count, so consecutive phases always
/// yield different verbs and a full cycle visits every verb.
pub(crate) fn verb_for_phase(seed: u64, phase: u64) -> &'static str {
    const STRIDE: u64 = 25;
    let count = SPINNER_VERBS.len() as u64;
    SPINNER_VERBS[((seed % count + (phase % count) * STRIDE) % count) as usize]
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn verb_for_phase_is_deterministic() {
        for seed in [0u64, 1, 42, u64::MAX] {
            for phase in [0u64, 1, 53, 54, 1000] {
                let verb = verb_for_phase(seed, phase);
                assert_eq!(verb, verb_for_phase(seed, phase));
                assert!(
                    SPINNER_VERBS.contains(&verb),
                    "{verb} should be a known verb"
                );
            }
        }
    }

    #[test]
    fn verb_for_phase_rotates_without_consecutive_repeats() {
        for seed in [0u64, 7, 42, 1337] {
            let cycle = (0..SPINNER_VERBS.len() as u64)
                .map(|phase| verb_for_phase(seed, phase))
                .collect::<HashSet<_>>();
            assert_eq!(
                cycle.len(),
                SPINNER_VERBS.len(),
                "cycle should visit every verb"
            );
            for phase in 0..2 * SPINNER_VERBS.len() as u64 {
                assert_ne!(
                    verb_for_phase(seed, phase),
                    verb_for_phase(seed, phase + 1),
                    "consecutive phases should differ for seed {seed}"
                );
            }
        }
    }

    #[test]
    fn verbs_are_unique_status_words() {
        let mut seen = HashSet::new();
        for verb in SPINNER_VERBS {
            assert!(verb.ends_with("ing"), "{verb} should end in ing");
            assert!(
                verb.chars().next().is_some_and(char::is_uppercase),
                "{verb} should start uppercase"
            );
            assert_ne!(verb, "Working");
            assert!(
                !verb.starts_with("Reviewing"),
                "{verb} should not collide with guardian review status"
            );
            assert!(seen.insert(verb), "{verb} should be unique");
        }
    }
}
