use super::*;
use pretty_assertions::assert_eq;

fn usage(input: i64, cached: i64, output: i64, reasoning: i64) -> TokenUsage {
    TokenUsage {
        input_tokens: input,
        cached_input_tokens: cached,
        output_tokens: output,
        reasoning_output_tokens: reasoning,
        total_tokens: input.saturating_add(output).saturating_add(reasoning),
    }
}

#[tokio::test]
async fn inherited_usage_is_not_charged_to_the_child() {
    let state = ChildTokenBudgetRuntimeState::new();
    state
        .initialize(
            usage(
                /*input*/ 10_000, /*cached*/ 2_000, /*output*/ 4_000,
                /*reasoning*/ 1_000,
            ),
            /*limit*/ 1_000,
        )
        .await;

    assert_eq!(
        ChildTokenBudgetAction::None,
        state
            .observe(&usage(
                /*input*/ 10_500, /*cached*/ 2_400, /*output*/ 4_100,
                /*reasoning*/ 1_100,
            ))
            .await
    );
}

#[tokio::test]
async fn cached_input_is_excluded_and_gemini_reasoning_is_included() {
    let state = ChildTokenBudgetRuntimeState::new();
    state
        .initialize(TokenUsage::default(), /*limit*/ 1_000)
        .await;

    assert_eq!(
        ChildTokenBudgetAction::Reminder {
            spent: 800,
            remaining: 200,
            limit: 1_000,
        },
        state
            .observe(&usage(
                /*input*/ 700, /*cached*/ 500, /*output*/ 300,
                /*reasoning*/ 300,
            ))
            .await
    );
}

#[tokio::test]
async fn reminder_fires_once_at_the_twenty_percent_remaining_boundary() {
    let state = ChildTokenBudgetRuntimeState::new();
    state
        .initialize(TokenUsage::default(), /*limit*/ 1_000)
        .await;

    assert_eq!(
        ChildTokenBudgetAction::None,
        state
            .observe(&usage(
                /*input*/ 799, /*cached*/ 0, /*output*/ 0, /*reasoning*/ 0,
            ))
            .await
    );
    assert_eq!(
        ChildTokenBudgetAction::Reminder {
            spent: 800,
            remaining: 200,
            limit: 1_000,
        },
        state
            .observe(&usage(
                /*input*/ 800, /*cached*/ 0, /*output*/ 0, /*reasoning*/ 0,
            ))
            .await
    );
    assert_eq!(
        ChildTokenBudgetAction::None,
        state
            .observe(&usage(
                /*input*/ 900, /*cached*/ 0, /*output*/ 0, /*reasoning*/ 0,
            ))
            .await
    );
}

#[tokio::test]
async fn exhaustion_supersedes_the_reminder_and_allows_one_grace_sample() {
    let state = ChildTokenBudgetRuntimeState::new();
    state
        .initialize(TokenUsage::default(), /*limit*/ 1_000)
        .await;

    assert_eq!(
        ChildTokenBudgetAction::BeginExhaustionGrace {
            spent: 1_000,
            limit: 1_000,
        },
        state
            .observe(&usage(
                /*input*/ 1_000, /*cached*/ 0, /*output*/ 0, /*reasoning*/ 0,
            ))
            .await
    );
    assert!(state.exhaustion_grace_in_flight().await);
    assert_eq!(
        ChildTokenBudgetAction::ForceCompletion {
            spent: 1_100,
            limit: 1_000,
        },
        state
            .observe(&usage(
                /*input*/ 1_100, /*cached*/ 0, /*output*/ 0, /*reasoning*/ 0,
            ))
            .await
    );
    assert!(!state.exhaustion_grace_in_flight().await);
    assert!(state.has_force_completed().await);
    assert_eq!(
        ChildTokenBudgetAction::ForceCompletion {
            spent: 1_200,
            limit: 1_000,
        },
        state
            .observe(&usage(
                /*input*/ 1_200, /*cached*/ 0, /*output*/ 0, /*reasoning*/ 0,
            ))
            .await
    );
}

#[tokio::test]
async fn zero_limit_disables_budget_actions() {
    let state = ChildTokenBudgetRuntimeState::new();
    state.initialize(TokenUsage::default(), /*limit*/ 0).await;

    assert_eq!(
        ChildTokenBudgetAction::None,
        state
            .observe(&usage(
                /*input*/ i64::MAX,
                /*cached*/ 0,
                /*output*/ i64::MAX,
                /*reasoning*/ i64::MAX,
            ))
            .await
    );
}

#[tokio::test]
async fn initialization_is_fixed_for_the_session_lifetime() {
    let state = ChildTokenBudgetRuntimeState::new();
    state
        .initialize(TokenUsage::default(), /*limit*/ 1_000)
        .await;
    state.initialize(TokenUsage::default(), /*limit*/ 10).await;

    assert_eq!(
        ChildTokenBudgetAction::None,
        state
            .observe(&usage(
                /*input*/ 100, /*cached*/ 0, /*output*/ 0, /*reasoning*/ 0,
            ))
            .await
    );
}
