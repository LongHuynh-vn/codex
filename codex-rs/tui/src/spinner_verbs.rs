//! Whimsical per-turn status verbs shown while the agent is working.

use rand::Rng;

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

pub(crate) fn random_verb<R: Rng + ?Sized>(rng: &mut R) -> &'static str {
    SPINNER_VERBS[rng.random_range(0..SPINNER_VERBS.len())]
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use pretty_assertions::assert_eq;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;

    #[test]
    fn random_verb_is_reproducible_for_seed() {
        let mut first = StdRng::seed_from_u64(42);
        let mut second = StdRng::seed_from_u64(42);

        let first_sequence = (0..8).map(|_| random_verb(&mut first)).collect::<Vec<_>>();
        let second_sequence = (0..8).map(|_| random_verb(&mut second)).collect::<Vec<_>>();

        assert_eq!(first_sequence, second_sequence);
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
