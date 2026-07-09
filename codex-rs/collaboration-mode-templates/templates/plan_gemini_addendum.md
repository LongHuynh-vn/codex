## Plan quality contract (strict)

These requirements extend the Finalization rule; where stricter, this contract wins.

For claims about the current code:

* Cite `path:line` for every claim about existing implementation, defaults, tests, schemas, persistence, or request shape.
* Use only facts verified during this session's exploration.
* Do not cite memory, prior runs, or unchecked assumptions as current source truth.
* If the request contradicts what the source does, flag that contradiction in the plan instead of planning around it.

For the final plan:

* End with a Critical Files section naming the 3-5 paths the change centers on.
* End with a Verification facts section naming 3-5 concrete facts the reviewer can check.
* Include a churn inventory for existing tests, snapshots, generated files, fixtures, and artifacts expected to change.
* If no existing tests, snapshots, or artifacts should change, state "none" explicitly.
* Search before saying "none"; do not infer it from memory.
* The `</proposed_plan>` closing tag ends your message; write nothing after it.

Keep plans compact and checkable:

* Every line should be either checkable against source, executable by the implementer, or needed to explain a decision.
* Cut narration, reassurance, and restatements of the user's prompt.
* Most good plans stay under 40 lines of prose, excluding paths, inventories, and requested detail.
* Small tasks still get small plans; this contract adds auditability, not length.

For verification:

* Name the exact tests or checks that should prove the change.
* Separate tests the implementer should run from tests the user or CI will own.
* Do not claim a test, build, search, or inspection was done unless it was actually done in this session.

For contradictions:

* If the source disagrees with the requested design, call that out before the implementation steps.
* Do not silently choose a workaround that changes the requested architecture.
* Flag the contradiction explicitly even when you already plan the correct behavior; a silent correction still counts as planning around it.
* When a cited fact becomes uncertain, re-open the source before finalizing the plan.

When the plan has 3 or more parts, or touches 3 or more files:

* Include a short risk register focused on the design's own seams.
* Mark each risk as VALIDATED or REJECTED with the source fact that decides the verdict.
* Keep the register short enough that it improves review rather than replacing the plan.
