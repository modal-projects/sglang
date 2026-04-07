# Disaggregated RL Policy Lifecycle Plan

Status: draft implementation plan for WIP development

Audience: engineers and coding agents implementing SGLang support for disaggregated RL trainers with low-bandwidth policy updates

## Purpose

This document is the implementation guide for extending SGLang into a flexible compatibility layer for disaggregated and online RL systems. It is intentionally developer-facing, not user-facing. It exists to keep implementation work aligned with the full vision even when individual subtasks are narrow or code complexity pushes the work toward local, short-sighted fixes.

This plan is opinionated. It is meant to be the guiding light for agents dropped into a mid-flight commit.

## Problem Statement

We want SGLang to support policy updates from external RL trainers when training and rollout inference are not co-located and high-bandwidth weight synchronization is unavailable or undesirable.

The feature matrix includes:

- Online LoRA inference, where the live policy is an unmerged LoRA overlay.
- Merged LoRA inference, where LoRA updates are merged into base weights for higher throughput but increased off-policy risk.
- Sparse full-parameter updates, where only the measurable finite-precision delta is communicated.
- Minimal-disruption update policies, including reuse of stale KV cache when the trainer accepts additional off-policiness.

The main mistake to avoid is forcing all of these cases into one low-level "weight apply" mechanism. The right core is policy lifecycle semantics with backend-specific capabilities.

## Core Decision

The core abstraction is **policy lifecycle**, not **prepared weight apply**.

The system should unify:

- Policy identity: alias vs concrete version
- Request/session policy pinning
- Activation policy
- Retirement policy
- Capability reporting
- Minimal provenance

The system should not unify the low-level apply path for:

- Online LoRA overlay serving
- Base-weight mutation serving

These are distinct backend families and should remain distinct:

### 1. Overlay Policy Backends

Used for online LoRA and any future lightweight overlay mechanism.

Properties:

- Multiple concrete versions may coexist in memory.
- Alias rebinding should be cheap.
- Drain-on-retire is desirable and practical.
- Requests can finish on the version they started with.
- Cache namespacing is already version-sensitive through resolved LoRA ids.

### 2. Mutation Policy Backends

Used for merged LoRA, sparse full-parameter updates, full checkpoint reload, and existing disk/tensor/distributed weight update paths.

Properties:

- Updating may mutate live model storage.
- Multiple concrete versions may not be able to coexist.
- Drain-on-retire is not assumed.
- Activation and cache policy must be explicit.
- Later, sparse prepared updates may use block-sparse manifests plus streamed value buckets.

## Full Vision

The end state should provide a policy lifecycle layer that sits above backend-specific update/application mechanisms.

### Common Lifecycle Concepts

- `PolicyAlias`
  A stable user-facing name such as `online`.
- `PolicyVersion`
  An immutable concrete policy instance.
- `PreparedUpdate`
  A target-local executable representation of an update for a specific backend.
- `ActivationPolicy`
  How a prepared version becomes visible to new requests.
- `RetirementPolicy`
  What happens to older concrete versions after activation.
- `CachePolicy`
  `flush`, `retract_recompute`, or `reuse_stale_kv`.
- `PolicyCapabilities`
  Backend-declared support matrix for activation, retirement, cache handling, coexistence, and pinning.

### Target Behavior

- Clients and routers refer to a stable alias, not mutable in-place state.
- Requests can optionally pin the concrete version they should use for the lifetime of a trajectory or session.
- New requests can resolve to a newer concrete version while stragglers finish on their old version when the backend supports drain-on-retire.
- The engine reports the concrete policy version used for a request. Richer provenance may come later, but it is not the architectural center of this work.
- Existing disk/tensor/distributed update paths become mutation backend implementations under the lifecycle layer, not parallel control planes with bespoke semantics.
- A future sparse full-parameter backend can prepare target-local block manifests and stream only changed value buckets without redefining the lifecycle layer.

## Current Foundation

The current codebase already provides useful building blocks, but they are not organized around policy lifecycle.

### Useful Existing Pieces

- LoRA already has immutable concrete ids and request-time acquire/release accounting.
- KV cache namespacing already depends on resolved LoRA id, which is the correct shape for overlay version isolation.
- LoRA updates are intentionally allowed to overlap inference.
- Existing mutation update paths already support multiple transport styles:
  disk, tensor, distributed group, IPC, and remote-instance loaders.
- `pause_generation` already exposes meaningful disruption modes:
  `abort`, `retract`, and `in_place`.

### Important Gaps

- There is no unified alias/version lifecycle surface.
- LoRA registry semantics are closer to "loaded adapter set" than "policy alias manager".
- Request/session pinning is not first-class.
- Mutation updates still couple update execution and cache disruption.
- Existing mutation paths are replacement-oriented, not sparse-delta oriented.
- Provenance is a single global `weight_version` string, which is not enough for future mixed-policy semantics.

## Non-Goals For The First Iterations

- Do not implement the block-sparse manifest backend first.
- Do not force online LoRA into the same low-level apply path as base-weight mutation.
- Do not make token-level mixed-version provenance the center of the architecture.
- Do not redesign external router/container-pool behavior inside SGLang.
- Do not require every backend to support drain-on-retire.

## Architectural Plan

## A. Introduce A Policy Lifecycle Layer

Create a small, explicit lifecycle layer whose job is to manage policy identity and visibility, not to own tensor movement.

Initial responsibilities:

- Resolve alias to concrete version
- Record backend capabilities
- Support activation and retirement transitions
- Support request/session pinning
- Expose clear errors when a requested semantic is unsupported by a backend

This layer must be small enough to unit test heavily and reason about in isolation.

## B. Refactor Online LoRA First

Online LoRA is the best first backend because the existing foundation already matches overlay lifecycle semantics.

Target end state for the LoRA path:

- Stable alias points to immutable concrete adapter version.
- Old concrete versions can remain resident after activation.
- Retirement policy is explicit:
  `evict_now`, `drain`, `drain_until_deadline`.
- Requests resolve aliases once and carry concrete version ids thereafter.
- If a session or trajectory is pinned, future requests in that session continue to use the same concrete version until explicitly released.
- New requests can resolve to the latest alias target without disturbing stragglers.

This is the backend where drain-on-retire is first-class and should be designed well, not bolted on.

## C. Put Existing Weight Update Paths Behind A Mutation Backend Interface

Do not rewrite their internals first. Wrap them in a mutation backend interface so the lifecycle layer can reason about them uniformly.

Initial mutation backend implementations:

- Existing disk update path
- Existing tensor update path
- Existing distributed group update path
- Existing IPC update path

Initial mutation backend semantics:

- Single live concrete version
- No drain-on-retire
- Explicit cache policy
- Optional prepare/activate split only where feasible without large memory spikes

## D. Prepare For Future Sparse Mutation Backend

Only after the lifecycle layer and backend split exist should we add a sparse full-parameter backend.

Its likely prepared form is:

- Block-sparse manifest
- Streamed value buckets
- Backend-specific shard/layout mapping
- Bounded-memory apply executor

This should plug into the mutation backend family, not redefine the architecture.

## Prioritized Implementation Plan

The phases below are ordered. Later phases may begin only after the earlier phase has crisp tests and acceptable invariants.

### Phase 1: Extract Policy Lifecycle Types And Capability Model

Goal:

- Add the lifecycle vocabulary without changing observable serving behavior.

Deliverables:

- New lifecycle types and enums:
  alias, concrete version id, backend family, activation policy, retirement policy, cache policy, capability matrix.
- Adapter layer that can describe current LoRA and current mutation paths without changing behavior yet.
- Minimal internal documentation in code comments where the semantics are not self-evident.

Primary code areas:

- `python/sglang/srt/managers/`
- likely a new policy-lifecycle module under `python/sglang/srt/`
- `python/sglang/srt/managers/io_struct.py`

Red-blue-refactor tests:

- Red:
  add pure unit tests for lifecycle types, capability checks, invalid combinations, and serialization behavior.
- Blue:
  implement the minimum type system and compatibility checks.
- Refactor:
  remove duplicated booleans or ad hoc policy strings where the new lifecycle types clearly replace them without changing behavior.

Test placement:

- `test/registered/unit/` mirroring the new lifecycle module location.

Exit criteria:

- No serving behavior changes.
- New lifecycle types are unit tested thoroughly.
- Existing RL and LoRA tests remain green unchanged.

### Phase 2: Refactor Online LoRA Into Alias/Version/Residency Semantics

Goal:

- Turn current dynamic LoRA loading into a true overlay policy backend.

Deliverables:

- Separate alias from concrete LoRA version.
- Residency manager for concrete LoRA versions.
- Retirement policies for overlay versions.
- Explicit alias activation operation.
- Refcount-aware retirement state machine.

Primary code areas:

- `python/sglang/srt/lora/lora_registry.py`
- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/lora/lora_manager.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/lora/lora_overlap_loader.py`

Red-blue-refactor tests:

- Red:
  start with unit tests that expose the intended registry state machine before modifying behavior.
  Add tests for:
  alias rebind,
  concrete version coexistence,
  acquire/release correctness,
  drain-on-retire,
  deadline-based forced retirement,
  residency budget enforcement,
  activation while an old version is still serving requests.
- Blue:
  implement the smallest registry/residency changes to make those tests pass.
- Refactor:
  move implicit assumptions out of `TokenizerManager` and into dedicated lifecycle or registry helpers.

Idiomatic net-new tests:

- Unit:
  registry state machine tests, residency budgeting tests, retirement policy tests, alias resolution tests.
- E2E:
  server tests where long-running requests keep using old LoRA while new requests resolve the newly activated version.
- E2E:
  verify cache isolation between concrete LoRA versions.
- E2E:
  verify that retirement does not break in-flight requests.

Test placement:

- unit tests under `test/registered/unit/` for state machines and registry logic
- E2E tests under `test/registered/lora/`

Exit criteria:

- Stable alias + concrete version semantics exist for LoRA.
- Drain-on-retire works and is proven by tests.
- New requests can pick up the latest alias target while old requests finish on older versions.

### Phase 3: Add Request And Session Policy Pinning

Goal:

- Make strict on-policy rollouts possible for online LoRA and future backends that support version coexistence.

Deliverables:

- Request-level or session-level concrete policy pinning
- Alias resolution once per pinned scope
- Explicit unpin/release behavior
- Clear errors when pinning is requested for a backend that cannot support it

Primary code areas:

- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/managers/io_struct.py`
- request/session management code under `python/sglang/srt/managers/`

Red-blue-refactor tests:

- Red:
  add failing E2E tests that show a trajectory or session should stay on the same concrete version even after alias activation.
  add failing negative tests where unpinned requests are expected to resolve fresh versions.
- Blue:
  implement minimum metadata plumbing and lifecycle hooks.
- Refactor:
  simplify request path resolution so policy pinning is obvious and centrally enforced.

Idiomatic net-new tests:

- Session-pinned rollout stays on old version while alias is rebound.
- Unpinned new request resolves newest version.
- Releasing a session allows future requests to resolve newest version.
- Backend capability mismatch raises a clean error.

Test placement:

- unit tests for pinning helpers
- E2E tests under `test/registered/lora/` and, if session-specific behavior is exercised, `test/registered/sessions/`

Exit criteria:

- Online LoRA can support strict on-policy rollouts through pinning.
- The faster "new requests get newest policy" behavior also works and is tested.

### Phase 4: Decouple Cache Policy From Mutation Update Transport

Goal:

- Replace ad hoc `flush_cache` booleans with explicit cache/disruption policy on mutation backends.

Deliverables:

- Explicit cache policy surface for mutation updates:
  `flush`,
  `retract_recompute`,
  `reuse_stale_kv`.
- Capability-based validation:
  if a backend cannot safely support a policy, it must fail clearly.
- Compatibility layer that maps current endpoints to the new semantics without immediate user-facing breakage.

Primary code areas:

- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/managers/tokenizer_communicator_mixin.py`
- `python/sglang/srt/managers/scheduler_update_weights_mixin.py`
- `python/sglang/srt/managers/scheduler.py`

Red-blue-refactor tests:

- Red:
  adapt existing RL weight-update tests to assert explicit policy behavior rather than only `flush_cache=True/False`.
  Add tests for invalid policy/backend combinations.
- Blue:
  implement policy plumbing with backward-compatible mapping.
- Refactor:
  remove duplicated update/flush branching where the new policy surface subsumes it.

Idiomatic net-new tests:

- `flush` fails while non-idle when appropriate.
- `retract_recompute` preserves requests and recomputes safely.
- `reuse_stale_kv` permits continuation when the backend declares support.
- Errors are deterministic and explicit for unsupported mutation backends.

Test placement:

- E2E tests under `test/registered/rl/`
- unit tests for policy validation under `test/registered/unit/`

Exit criteria:

- Mutation update behavior is controlled by explicit cache policy.
- Existing RL update tests are still covered and stronger than before.

### Phase 5: Wrap Existing Weight Update Paths In Mutation Backend Interface

Goal:

- Create a clean backend seam without changing transports yet.

Deliverables:

- Mutation backend interface
- Current disk/tensor/distributed/IPC implementations behind the interface
- Capability reporting for prepare/activate, coexistence, drain, and cache policies

Primary code areas:

- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/managers/tp_worker.py`
- update request plumbing under `python/sglang/srt/managers/`

Red-blue-refactor tests:

- Red:
  unit tests for capability declarations and backend selection
  E2E coverage reusing current RL tests to ensure no regression
- Blue:
  add the interface and adapt current paths
- Refactor:
  reduce transport-specific logic leaking into higher layers

Exit criteria:

- Lifecycle layer can reason about mutation backends uniformly.
- No transport rewrite yet.

### Phase 6: Add Minimal Concrete Policy Provenance

Goal:

- Report truthful concrete policy identity with minimal performance overhead.

Deliverables:

- Concrete version id in request metadata
- For streaming, full summary only on final response if additional summary fields are introduced later

Important constraint:

- Do not let provenance design dominate the core architecture.
- Keep this phase intentionally modest.

Tests:

- E2E tests that assert the reported concrete version matches the version actually pinned/resolved for the request
- No speculative token-level accounting in this phase

### Phase 7: Future Sparse Mutation Backend

Goal:

- Add low-bandwidth sparse full-parameter mutation support after the lifecycle split is stable.

Deliverables:

- Prepared sparse update representation
- Logical-to-local parameter layout registry
- Block manifest plus streamed value bucket executor
- Bounded-memory apply path
- Validation and rollback semantics

Important note:

- This phase is intentionally later.
- It should not begin before the overlay backend and mutation backend split are stable.

## Testing Strategy

Testing must be rigorous. Do not cut corners because the control flow is complex or because server tests are expensive.

### Testing Principles

- Start with existing tests where they already encode a real contract.
- Add new tests first for any new lifecycle or backend semantic.
- Prefer unit tests for state machines, capability matrices, alias resolution, residency policies, and retirement logic.
- Use E2E tests only for process boundaries, request lifetimes, cache interactions, and cross-component behavior.
- Negative tests are mandatory for unsupported capability combinations.
- Race-sensitive behavior must be tested with explicit synchronization or deterministic structure, not timing-only assertions where avoidable.
- Avoid goldens based only on log strings.
- Avoid tests that prove only the happy path.
- Avoid reducing assertions just to make brittle code pass.

### Required Test Layers

#### Unit Tests

Use for:

- lifecycle type validation
- alias/version registry state machine
- retirement policies
- capability checks
- request/session pinning helpers
- mutation backend selection and policy validation

Placement:

- `test/registered/unit/` mirroring source tree

#### E2E Server/Engine Tests

Use for:

- LoRA alias activation while requests are in flight
- drain-on-retire behavior
- session pinning across multiple requests
- explicit cache policy behavior for mutation updates
- non-regression of existing disk/tensor/distributed update paths

Placement:

- `test/registered/lora/`
- `test/registered/rl/`
- `test/registered/sessions/` when session-specific semantics are involved

### Red-Blue-Refactor Discipline

For every phase:

1. Red
   Write or adapt the smallest rigorous failing test set that captures the new contract.
2. Blue
   Implement the smallest change that makes the tests pass without broad speculative refactors.
3. Refactor
   Clean up code structure only after behavior is locked by tests.

Do not skip the red phase by writing tests after the code.

## Concrete Test Roadmap

The following sequence should guide implementation.

### First Test Additions

- unit test for lifecycle capability validation
- unit test for alias rebind semantics
- unit test for drain-on-retire state machine
- unit test for version residency budget behavior
- unit test for request/session pinning semantics

### First E2E Additions

- LoRA alias switch while long-running requests are active
- new requests resolve latest alias while old requests finish on old concrete version
- pinned session remains on old concrete version after alias switch
- explicit retirement deadline eventually evicts idle old version
- cache remains isolated between concrete LoRA versions

### Mutation Path E2E Strengthening

Adapt existing RL tests so they continue to verify:

- update from disk
- update from tensor
- update from distributed group
- pause/continue semantics

Then strengthen them to verify:

- explicit cache policy behavior
- unsupported policy/backend combinations
- no silent fallback from one policy to another

## Guidance For Agents

When working from this plan:

- Keep the big picture in mind: lifecycle first, backend apply second.
- Do not unify online LoRA with base-weight mutation at the low-level apply layer.
- Do not start on the sparse mutation backend before the lifecycle split is stable.
- Prefer small, test-anchored refactors over broad rewrites.
- If a subtask seems to require forcing overlay and mutation semantics together, stop and revisit the architecture.
- When in doubt, preserve backend differences and unify only the semantics that matter to RL integrations.

## Definition Of Done For The Overall Effort

The effort is not done when a single new endpoint exists. It is done when:

- Online LoRA behaves as a real versioned overlay backend.
- Strict on-policy and high-throughput fresh-request modes are both supported and tested.
- Existing mutation transports sit behind a mutation backend interface.
- Cache/disruption policy is explicit and backend-validated.
- The code structure makes room for a later sparse mutation backend without another architectural reset.
- Tests are strong enough that future agent work can extend the system without re-deriving the design from scratch.
