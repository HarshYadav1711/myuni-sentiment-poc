# Independent wellbeing / context classifier (Phase 4 foundation)

## Purpose

This document describes the **independent wellbeing / context classifier**
added as a Phase 4 foundation. It classifies **content-level** wellbeing
relevance, target, and expressed language signals.

It is **not** wired into the existing temporal wellbeing gate,
`FinalTemporalAssessment`, OpenRouter `context_type`, or the Gradio client
yet. That wiring is a later milestone.

## Sentiment ≠ wellbeing

The existing pipeline scores **sentiment** (positive / neutral / negative)
from text, speech, and visual evidence.

**Negative sentiment does not mean poor wellbeing.**

Examples:

- “This movie was depressing.” → negative media sentiment, not personal wellbeing.
- “Our university hosts a mental health seminar.” → wellbeing *topic*, not the author’s state.
- “My friend is stressed.” → wellbeing-related language about *someone else*.

Personal wellbeing relevance and target must therefore be decided by an
**independent classifier**, not inferred from sentiment polarity or from
OpenRouter narrative `context_type`.

## Why an independent classifier

Today, personal-wellbeing gating in the POC is effectively tied to the
reasoner’s `context_type`. That couples a transparent rule gate to an LLM
narrative label.

Phase 4 separates the concerns:

1. **Sentiment / temporal evidence** remain the protected baseline.
2. **Wellbeing relevance / target / expressed signals** become their own
   content-level classification task.
3. Later, the wellbeing gate can consume classifier outputs instead of
   treating OpenRouter as the authority for those labels.

## Model choice (POC)

**Model ID:** `MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c`

Rationale:

- NLI-based zero-shot classification via Hugging Face Transformers
- CPU-capable for local development
- Supports exclusive (`multi_label=False`) and multi-label (`multi_label=True`) heads
- MIT model license
- Commercially friendlier **`-c`** training-data variant
- No paid API, no credit-card dependency, no OpenRouter usage
- Can be benchmarked or replaced later without changing result schemas

Inference is **local only**. No environment variable may silently turn this
classifier into a remote inference API call.

### Why the `-c` variant

The `-c` checkpoint is the commercially-oriented training-data variant of
the same zero-shot DeBERTa family. For a university product POC that may
later need redistribution or commercial evaluation, preferring `-c` reduces
licensing ambiguity versus non-`-c` counterparts while keeping the same
Transformers API.

## Taxonomies

All candidate verbalizations and hypothesis templates live in
`src/wellbeing/labels.py` (no scattered hidden prompts).

### Relevance (exclusive)

| Label | Meaning |
| --- | --- |
| `personal_wellbeing` | Author expresses their own current/recent wellbeing, stress, difficulty, coping, or recovery |
| `wellbeing_topic_only` | Discusses wellbeing topics without clearly stating the author’s own state |
| `not_wellbeing_related` | No meaningful personal-wellbeing content |
| `ambiguous` | Text is insufficient or unclear |

### Target (exclusive)

| Label | Meaning |
| --- | --- |
| `self` | Substantially about the speaker/author |
| `other_person` | About another referenced person |
| `group_or_community` | About a group/community |
| `institution_or_event` | About an institution, event, news/media item, class, university, etc. |
| `general_or_unknown` | Target cannot be reliably determined |

### Expressed signals (multi-label)

These names describe **language / evidence in content**. They are **not**
diagnoses.

- `stress_or_overwhelm`
- `anxiety_or_fear_language`
- `loneliness_or_isolation`
- `hopelessness_like_language`
- `exhaustion_or_burnout_like_language`
- `self_directed_negativity`
- `interpersonal_distress`
- `academic_pressure`
- `positive_wellbeing_or_recovery`

Example: `anxiety_or_fear_language` means the text *uses* anxiety/fear-like
wording. It does **not** mean “the user has an anxiety disorder.”

## Safety boundaries

The classifier must **not**:

- Diagnose mental-health conditions
- Infer a person’s hidden mental state
- Infer wellbeing from facial appearance
- Use SigLIP / facial cues as classifier inputs
- Use sentiment score as a substitute for wellbeing relevance
- Expose labels such as `depressed`, `depression`, `mentally_ill`,
  `suicidal_person`, `anxiety_disorder`, `bipolar`, `PTSD`

## Missing ≠ neutral

`None`, empty, whitespace-only, and extremely short text return
`status=insufficient_text` **without model inference**.

The system must not invent neutral wellbeing evidence for missing inputs.

## Scores are not clinical probabilities

Zero-shot NLI scores are **model evidence**. They are:

- not calibrated clinical probabilities
- not a 0–10 or 0–100 “mental health score”
- not a wellbeing score of any kind

### Provisional signal threshold

`WELLBEING_SIGNAL_POC_THRESHOLD` (default `0.5`) controls
``threshold_passed`` on each signal.

**This threshold is provisional and must be calibrated against an in-domain
validation set.** It is not clinically validated.

### Self-attribution dual evidence (Phase 4A.5)

Production attribution uses **two independent** zero-shot heads with
``multi_label=True``:

1. ``direct_self_experience`` — author describing their own wellbeing
2. ``reported_other_experience`` — quoted / reported / other-person experience

Policy derives ``self_experience`` / ``not_self_experience`` / ``unclear``:

- self when ``direct_self >= DIRECT_SELF_MIN_SCORE`` and
  ``reported_other < REPORTED_OTHER_BLOCK_SCORE``
- not-self when reported-other meets the block score and direct-self is
  insufficient
- unclear on conflict (both strong) or insufficient evidence

**Mixed self+other:** strong evidence on both heads prefers ``unclear``
rather than auto-blocking solely because other-person evidence exists.

Legacy exclusive binary attribution (``multi_label=False``) is retained for
evaluation comparison only and must not drive eligibility.

``selected=true`` only when ``eligibility_status == eligible`` AND
``threshold_passed``.

**Dataset roles:**

- ``calibration_cases.py`` — development/calibration (fit policy)
- ``attribution_dev_cases.py`` — attribution-focused development coverage
- A–K fixtures — regression/development (contaminated)
- ``final_holdout_cases.py`` (FH40) — NOW also development/regression
  (failures influenced this design; not unbiased)
- ``final_holdout_v2.py`` — fresh holdout, evaluated once after freeze

Thresholds (frozen from development calibration, Phase 4A.5):

- ``WELLBEING_RELEVANCE_MIN_MARGIN`` = ``0.0``
- ``WELLBEING_TARGET_MIN_MARGIN`` = ``0.0``
- ``WELLBEING_DIRECT_SELF_MIN_SCORE`` = ``0.35``
- ``WELLBEING_REPORTED_OTHER_BLOCK_SCORE`` = ``0.55``

Conflict rule: if ``direct_self >= self_min`` AND
``reported_other >= other_block`` → ``unclear`` (prefer abstention).

### Conservative personal selection

Raw scores are always retained.

``selected=true`` only when:

1. ``threshold_passed``
2. ``eligibility_status == eligible``

Topic-only / other-person / quoted-other cases may still show high raw
signal scores, but those signals are **not** selected personal-wellbeing
evidence.
### Batched classification

``WellbeingClassifier.classify_many(texts)`` runs relevance, target, and
signals over usable inputs, with ``WELLBEING_CLASSIFIER_BATCH_SIZE``
(default 4) for CPU memory safety. Dual attribution runs only for
``personal_wellbeing`` + ``self`` candidates. ``classify()`` remains a thin
wrapper.

## Config

| Setting | Role |
| --- | --- |
| `WELLBEING_CLASSIFIER_ENABLED` | Enable/disable classifier (disabled → `classifier_unavailable`) |
| `WELLBEING_CLASSIFIER_MODEL` | Local HF model id (default DeBERTa `-c`) |
| `WELLBEING_SIGNAL_POC_THRESHOLD` | Provisional multi-label selection floor |

Remote API URLs are rejected as model ids.

## Runtime behavior

- Lazy-load: importing `src.wellbeing` does **not** download or load the model
- Load once, reuse, CPU by default, eval / no-grad path via Transformers pipeline
- Deterministic intent: temperature not used; exclusive/multi-label heads are NLI scoring
- Fail-soft statuses: `insufficient_text`, `classifier_unavailable`, `error`
- Errors must not copy raw user text into logs or diagnostics

API sketch:

```python
from src.wellbeing import WellbeingClassifier

clf = WellbeingClassifier()
clf.load()                      # optional explicit load
result = clf.classify(text, source_type="text")
assert clf.is_loaded
```

## Package layout

Schemas live in `src/wellbeing/schemas.py` (package-local), not in the
pipeline-wide `src/schemas.py`, so this foundation stays isolated until a
later wiring pass — same pattern as `src/temporal/benchmark/schemas.py`.

## Phase 4B — shadow-mode pipeline integration

Shadow mode attaches the independent classifier to real analysis flows
**without** giving it authority over client-facing decisions.

**Negative sentiment is not equivalent to poor wellbeing.**

**Shadow evidence does not alter the client-facing wellbeing indicator.**

### What shadow mode is

When `WELLBEING_SHADOW_ENABLED=true`:

1. Collect authored textual evidence already produced by the pipeline
2. Run DeBERTa via `classify_many` (batched)
3. Store `AnalysisBlock.wellbeing_shadow` (`WellbeingShadowAnalysis`)

When disabled (default):

- `wellbeing_shadow` remains `None`
- DeBERTa is **not** loaded or downloaded

### Parallel architecture

```
                 ┌─ sentiment / fusion / temporal / OpenRouter / gate
content ─────────┤
                 └─ DeBERTa wellbeing shadow ──→ wellbeing_shadow only
```

No authority crossover in Phase 4B.

### Inputs that feed the shadow classifier

| Source | Used? | Role |
| --- | --- | --- |
| Primary text activity | Yes | `primary_text` |
| Author-provided caption | Yes | `caption` |
| Faster-Whisper transcript | Yes (reuse only) | `transcript` |
| Usable temporal window speech | Yes (reuse segments) | `speech_window` |
| Image/video OCR | **No** | uncertain authorship |
| SigLIP / faces / frames | **No** | safety boundary |

### Why OCR is excluded

OCR may come from memes, slides, quotes, subtitles, or reposted video.
Until a provenance policy exists, OCR must not become personal wellbeing
evidence.

### Why ASR / timestamps are reused

Shadow mode reuses:

- the existing ONE Faster-Whisper pass (`bundle.transcript`)
- existing `TemporalContext` windows and `speech_segments`

It does **not**:

- rerun Whisper
- recompute 5-second windows
- modify temporal sentiment

### Window outputs

Per-window shadow results store index / start / end / classification only.
**No raw speech text** is persisted in `wellbeing_shadow`.

### CPU latency / default OFF

DeBERTa CPU inference is expensive. Shadow mode defaults to **OFF** so
deployment and ordinary analysis do not silently add large runtime.

Enable explicitly for local/evaluation runs via `WELLBEING_SHADOW_ENABLED`.

### Failure isolation

If the classifier fails to load or infer, the main analysis still succeeds
and `wellbeing_shadow.status` records `error` / `classifier_unavailable`.

### What comes next (Phase 4C)

Use shadow evidence from controlled live runs to decide whether (and how)
classifier outputs may later inform the wellbeing gate — still without
conflating sentiment polarity with wellbeing.

## Phase 4C.1 — deterministic wellbeing evidence aggregation

Phase 4C.1 adds a **pure post-processing evidence feature layer** over
already-produced Phase 4B shadow classifications.

- **No new model inference**
- **No concern category** (`low_concern` / `moderate_concern` / `high_concern`)
- **No numerical wellbeing / mental-health score**
- **Does not replace** `compute_wellbeing_indicator()` or change
  `overall_wellbeing_indicator`
- Attached optionally as `WellbeingShadowAnalysis.evidence_context`

Implementation: `src/wellbeing/evidence.py` →
`build_wellbeing_evidence_context(...)`.

### Global vs local evidence roles

Global transcript / primary text / caption and local speech windows are
**not equivalent votes**.

| Role | Purpose |
| --- | --- |
| **Global** | Authorship, context, whole-content semantics |
| **Windows** | Temporal localization, recurrence, transitions |

Do **not** majority-vote `1 global + N windows` as `N+1` equal ballots.
Do **not** average relevance scores or signal probabilities across sources.

### Why windows are not independent votes

A five-second fragment and a full transcript answer different questions.
Windows measure **where / how often / whether runs recur**. Global
classification measures **overall authored meaning**. Treating them as
votes collapses that distinction.

### Why signal probabilities are not averaged

Selected-signal **presence across independently classified eligible
windows** is the temporal evidence. Averaging raw DeBERTa scores would
invent a pseudo-severity the model does not provide.

### Temporal recurrence vs severity

Eligible-run length and `eligible_window_fraction` are **descriptive
recurrence features**, not severity. Phase 4C.1 does **not** label a run
“persistent distress” merely because it has length N. Policy semantics
belong in Phase 4C.2.

### Recovery evidence

`positive_wellbeing_or_recovery` is first-class. Recovery windows are
counted separately from distress-like language signals. Mixed eligible
windows (recovery **and** distress-like selected) are counted as
`mixed_signal_window` — signals are preserved, not averaged away.

### Conflicts / diagnostics

Machine-readable disagreement codes only (not resolved yet), e.g.:

- `global_eligible_no_local_support`
- `local_eligible_without_global_eligibility`
- `global_recovery_local_distress`
- `global_distress_local_recovery`

### Why visual sentiment is excluded

Faces / SigLIP / frame sentiment are **not** personal-wellbeing evidence.
Observable visual negativity must not silently become inferred internal
wellbeing.

### Why multimodal temporal negativity is excluded

`TemporalFeatures.negative_persistence`, `trajectory`, and combined
`negative_probability` remain **sentiment / temporal context**. They must
**not** become hidden inputs to wellbeing policy. Phase 4C.1 reads only
`WellbeingShadowAnalysis` classification fields (plus window
index/start/end provenance).

### Why no concern level in 4C.1

Phase 4C.1 ships an inspectable evidence context only. Concern policy is
Phase 4C.2 and remains shadow-only until Phase 4C.3 validation.

### Input to Phase 4C.2

Phase 4C.2 consumes `WellbeingEvidenceContext` (global + temporal
evidence, eligible runs, signal recurrence, recovery/distress/mixed
counts, conflict diagnostics) — not raw multimodal sentiment features.

## Phase 4C.2 — deterministic shadow wellbeing policy candidate

Phase 4C.2 adds a **transparent deterministic policy candidate** on top of
`WellbeingEvidenceContext`.

Implementation: `src/wellbeing/policy_candidate.py` →
`build_wellbeing_policy_candidate(evidence)`.

Attached as `WellbeingShadowAnalysis.policy_candidate`.

**Still shadow-only.** It does **not** replace:

- `src/temporal/wellbeing.py` / `compute_wellbeing_indicator()`
- `FinalTemporalAssessment` / `build_final_temporal_assessment()`
- Gradio client wellbeing result
- routed analysis status

Policy version: `phase4c2-v1`.

### Evidence vs policy separation

| Layer | Object | Role |
| --- | --- | --- |
| 4C.1 evidence | `evidence_context` | Deterministic facts |
| 4C.2 policy | `policy_candidate` | Deterministic interpretation |

Do not mix these concepts. Facts stay inspectable even when the candidate
abstains.

### Global contextual authority

GLOBAL transcript/content evidence has more linguistic context than short
fragments. Categorical concern candidates normally require **globally
eligible** personal wellbeing evidence.

WINDOWS provide temporal/local support only. They do **not** promote
local-only fragments into low/moderate/high.

### Window temporal-support role / why windows are not votes

Windows answer localization and recurrence. They are not ballots equal to
the global source. No majority vote. No averaging of signal scores.

### Candidate indicator semantics

| Indicator | Meaning (content-level) |
| --- | --- |
| `low_concern` | Global eligible recovery-only / no distress, without recurrent/persistent local distress |
| `moderate_concern` | Global eligible distress; local distress none/isolated/recurrent (local support **not** required) |
| `high_concern` | Global eligible distress **and** persistent local distress (≥2 consecutive distress-supporting windows) |
| `insufficient_evidence` | Gate failure, local-only, or unresolved mixed conflict |

These are **not** diagnoses, clinical risk levels, suicide-risk predictions,
mental-health scores, or medical triage.

### Why high does not mean clinical high risk

`high_concern` means persistent **content-level** personal distress evidence
under a transparent engineering recurrence rule. It is not clinical high
risk and must not be presented as such.

### Why distress signal names have no severity weights

Signal identity alone does not determine concern level. There is no encoding
such as `hopelessness = 3`, `anxiety = 2`, `stress = 1`.

### Why isolated “hopelessness-like” does not force high

Isolated local distress (including `hopelessness_like_language` or
`self_directed_negativity` alone) yields at most **moderate** when global
distress exists. High requires **temporal persistence** (consecutive
distress-supporting windows), not a particular signal name.

### Why mixed recovery/distress abstains

Contradictory recovery and distress evidence is not averaged. Material
unresolved conflicts (e.g. global recovery + persistent local distress,
global distress + recurrent local recovery, global distress+recovery)
return `status=conflict` and `indicator=insufficient_evidence`.

### Why local-only evidence cannot override global context

Short speech fragments can lose attribution/context. If global is not
eligible / uncertain / absent, local eligible windows remain diagnostic
only (`local_eligible_without_global_eligibility`) and cannot produce
low/moderate/high.

### Why global-only eligible evidence remains usable

Global eligible distress with zero local support may still be
`moderate_concern`. Global eligible recovery with no local distress may be
`low_concern`. Diagnostic `global_eligible_no_local_support` is preserved
but does not invalidate the candidate.

### Exclusions (mandatory input boundary)

Policy consumes **only** `WellbeingEvidenceContext`. It must not inspect:

- raw transcript / speech / caption text
- OCR
- SigLIP / visual sentiment / faces
- sentiment probabilities
- `TemporalFeatures.negative_persistence` / `trajectory`
- `TemporalWindow.negative_probability`
- multimodal cross-modal conflict
- OpenRouter reasoning
- current overall wellbeing indicator

### POC nature of 2-consecutive-window persistence

`persistent` distress = **≥ 2 consecutive** eligible windows containing
selected distress-like evidence.

This is a **transparent engineering recurrence rule**, not clinically
validated persistence.

### Phase 4C.3

Validate this shadow candidate against controlled evidence before any
authority wiring into the client-facing wellbeing path.

## Phase 4C.3 — deterministic policy validation

Phase 4C.3 validates `phase4c2-v1` with a **deterministic structured
scenario harness**. It does **not** change policy authority.

Implementation:

- `evaluation/wellbeing/policy_scenarios.py` — human-authored evidence
  fixtures + expected outcomes (independent of the policy function)
- `evaluation/wellbeing/policy_validation.py` — engineering validation
  harness / metrics / invariants
- `tests/test_wellbeing_policy_validation.py` — pytest entrypoint

### Structured scenario validation

Scenarios are synthetic **engineering evidence states**, not student
examples and not clinical data. Each fixture includes explicit expected:

- status / indicator
- local_support_level / distress_pattern / recovery_pattern
- required reason codes

Expected values are **not** derived by calling the policy.

### Policy invariants (engineering safety)

1. No globally eligible personal wellbeing → never low/moderate/high
2. High requires global eligible distress **and** ≥2 consecutive
   distress-supporting windows
3–5. Isolated hopelessness / self-negativity / anxiety alone never force high
6. Different distress labels across adjacent windows may still form
   persistent temporal support
7. Recovery does not simply subtract from distress
8. Material mixed recovery/distress conflict is never averaged into a
   fake middle category
9. Local windows cannot override globally non-personal attribution
10. Global personal distress remains usable when local windows lack context

### Temporal recurrence semantics

Adjacency uses **window_index**, not timestamp inference. A short final
window can still complete a consecutive pair. Nonconsecutive indices
(`[1,3]`) do not form persistence.

### Recovery / distress conflict semantics (phase4c2-v1)

| Case | Result |
| --- | --- |
| Global recovery + no local distress | `low_concern` |
| Global recovery + recovery windows | `low_concern` |
| Global recovery + isolated local distress | `low_concern` |
| Global recovery + persistent local distress | `conflict` / `insufficient_evidence` |
| Global distress + recurrent local recovery | `conflict` / `insufficient_evidence` |
| Global distress + recovery (mixed global) | `conflict` / `insufficient_evidence` |

### Controlled-video structured replay

One fixture (`replay_controlled_video_phase4b`) replays the previously
observed Phase 4B/4C structured evidence shape (global distress + isolated
window-3 distress including hopelessness-like language). Expected:
`moderate_concern`. This is **structured replay**, not a new live run.

### Adversarial scenarios

Corpus includes attempts such as all eight distress IDs in one window,
many nonconsecutive distress windows, local-only persistence, missing
evidence IDs, classifier-error populated fields, and out-of-order window
construction. Policy must fail safe / remain deterministic.

### Engineering validation — not clinical validation

Reported metrics are engineering agreement / invariant / determinism /
evidence-grounding checks. They are **not** clinical accuracy,
sensitivity, specificity, or medical validation.

### No ML thresholds changed

Phase 4C.3 does not retune DeBERTa, attribution thresholds, signal
threshold, Whisper, SigLIP, OpenRouter, or sentiment/temporal formulas.

### Policy remains shadow-only

Even after validation passes, `affects_final_assessment=False`. Do **not**
wire into `FinalTemporalAssessment` or replace
`compute_wellbeing_indicator()` in this phase. Next step:
Phase 4C.4 shadow replay before any authority migration.

## Phase 4C.4 — end-to-end shadow chain replay

Phase 4C.4 validates the **full shadow wellbeing chain** by replaying
structured Phase 4B-style classifier outputs through production builders:

1. structured source/window classification results
2. `build_wellbeing_evidence_context(...)`
3. `build_wellbeing_policy_candidate(...)`

Implementation:

- `evaluation/wellbeing/shadow_replay_cases.py`
- `evaluation/wellbeing/shadow_replay.py`
- `tests/test_wellbeing_shadow_replay.py`

### No model inference

Replay does **not** run DeBERTa, Whisper, SigLIP, OCR, OpenRouter, Qwen,
or video inference. Fixtures contain structured classification states only
(no raw transcript/caption/speech text).

### Production builders are used

Expected fixture values are human-authored. The harness invokes the real
evidence and policy functions to validate composition — fixtures do not
precompute `WellbeingEvidenceContext` / `WellbeingPolicyCandidate`.

### Modality provenance boundaries

| Modality | Allowed global source | Local |
| --- | --- | --- |
| Text | `primary_text` | none |
| Audio | `transcript` | none |
| Caption/image authored | `caption` | none |
| Video | `transcript` | `speech_window` |

OCR / visual / SigLIP / faces are **never** personal wellbeing sources.

### Privacy checks

Serialized `WellbeingShadowAnalysis`, `WellbeingEvidenceContext`, and
`WellbeingPolicyCandidate` payloads are recursively inspected for forbidden
raw-text keys (`text`, `transcript_text`, `speech_segments`, etc.).

### Deterministic chain validation

Each case runs twice; outputs must match and inputs must not mutate.
Evidence IDs used by the policy must be grounded in
`WellbeingEvidenceContext.evidence_ids`.

### Old temporary gate is not ground truth

The existing `src/temporal/wellbeing.py` gate uses different evidence and
must **not** be treated as the scoring oracle for the new shadow policy.
Disagreement with the old gate is diagnostic only, not an automatic
new-policy error.

### Candidate remains shadow-only

`affects_final_assessment=False` for evidence and policy. Phase 4C.4 does
not wire into `FinalTemporalAssessment` or Gradio.

### Next step

Phase 4C.5 should be **authority-migration design** — not automatic
replacement of the client-facing wellbeing indicator.

## Phase 4C.5 — authority migration design (control plane)

Phase 4C.5 adds the **migration control plane** for eventually making
`phase4c2-v1` authoritative on **VIDEO** wellbeing assessment.

It does **not** activate candidate authority.

Implementation:

- `src/wellbeing/migration.py` → `compare_wellbeing_policies(...)`
- `WellbeingAuthorityComparison` schema (attached to shadow only)
- `WELLBEING_AUTHORITY_MODE` / `WELLBEING_CANDIDATE_AUTHORITY_ACTIVATED`
- `tests/test_wellbeing_migration.py`

### Authority modes

| Mode | Meaning |
| --- | --- |
| `legacy` (default) | `src/temporal/wellbeing.py` remains authoritative |
| `compare` | Legacy authoritative; candidate compared; diagnostics recorded |
| `candidate` | Reserved — **not activated** in Phase 4C.5 |

Default is **legacy**. Candidate mode requires a later activation phase
(`WELLBEING_CANDIDATE_AUTHORITY_ACTIVATED=true`) and must not silently
route `FinalTemporalAssessment`.

### Why compare does not alter client output

Compare mode records `authority_comparison` on `WellbeingShadowAnalysis`
only. `FinalTemporalAssessment.overall_wellbeing_indicator` remains the
legacy value. `affects_final_assessment=False`.

### Technical failure vs semantic abstention

| Kind | Examples |
| --- | --- |
| Technical failure | shadow disabled, classifier unavailable/error, policy unavailable |
| Semantic abstention | globally non-personal, attribution uncertain, recovery/distress conflict, insufficient personal evidence |

`insufficient_evidence` from quoted-other / topic-only content is
**successful conservative behavior**, not a system failure.

### Fallback contract

- Compare / legacy: legacy remains authoritative; no runtime fallback action
- Candidate mode (future, when activated): technical failure may fall back
  to legacy with explicit `fallback_reason`
- Semantic abstention must **not** auto-fallback unless a future contract
  explicitly chooses that

### Rollback

Rollback requires only:

```text
WELLBEING_AUTHORITY_MODE=legacy
```

No code revert. `src/temporal/wellbeing.py` remains intact.

### Shadow / authority configuration matrix

| Shadow | Authority mode | Behavior |
| --- | --- | --- |
| False | legacy | Existing behavior only |
| True | legacy | Existing authority + shadow evidence (+ optional diagnostics) |
| True | compare | Existing authority + candidate + comparison diagnostics |
| False | compare | Candidate unavailable; legacy remains authoritative |
| * | candidate | Reserved / not activated in 4C.5 (`candidate_mode_not_activated`) |

`WELLBEING_SHADOW_ENABLED` and `WELLBEING_AUTHORITY_MODE` are independent.
Compare mode does **not** silently enable shadow.

### Disagreement taxonomy

Examples: `legacy_low_candidate_moderate`,
`legacy_moderate_candidate_high`, `candidate_conflict`,
`candidate_unavailable`. **Different ≠ candidate better.**

### Future UI terminology (not implemented yet)

Avoid “Overall Well-Being Score” / numerical score language.

Recommended direction:

- Label: **Wellbeing Indicator**
- Categories: **Low Concern / Moderate Concern / High Concern /
  Insufficient Evidence**

Avoid clinical/risk phrasing (diagnosis, suicide risk, emergency triage).

### High Concern meaning

**High Concern** means persistent content-level personal distress evidence
under the POC recurrence policy (≥2 consecutive distress-supporting
windows). It does **not** mean high suicide risk, severe mental illness,
clinical emergency, or diagnosis.

### Video-only migration scope

Authority migration is scoped to **VIDEO** `FinalTemporalAssessment`.
Text / image / audio may keep shadow/evaluation evidence without becoming
authoritative product outputs in this phase.

### Migration-readiness checklist (future activation)

Before candidate becomes authoritative, require:

- [x] 4C.3 deterministic policy validation passed
- [x] 4C.4 full shadow-chain replay passed
- [ ] real compare-mode video validation
- [ ] disagreement review completed
- [ ] runtime acceptable
- [ ] failure/fallback validated in live compare
- [ ] rollback tested operationally
- [ ] UI terminology approved
- [ ] client/product semantics approved
- [ ] observability exists

### Candidate mode not activated yet

`WELLBEING_CANDIDATE_AUTHORITY_ACTIVATED` defaults to **false**.
Phase 4C.6 should perform controlled live compare validation before any
authority activation design sign-off.

## Why OpenRouter is not the authority for classifier labels

OpenRouter is used for **contextual temporal reasoning** over structured
evidence. That is a different job from content-level wellbeing taxonomy
labels.

- Reasoner output can narrate uncertainty and cite evidence ids
- Classifier output should own relevance / target / expressed-signal labels
- Keeping them separate prevents LLM narrative drift from silently becoming
  the wellbeing ontology

## Limitations (current pass)

- Zero-shot DeBERTa is a POC baseline, not an in-domain trained classifier
- Signal threshold remains provisional (0.5)
- Shadow mode is non-authoritative (Phase 4B) — does not feed
  `compute_wellbeing_indicator()` or `FinalTemporalAssessment`
- Evidence aggregation (Phase 4C.1) is non-authoritative and produces no
  concern category by itself
- Policy candidate (Phase 4C.2) is shadow-only; 2-window persistence is an
  engineering POC rule, not clinically validated
- Phase 4C.3 validates the candidate with structured scenarios only;
  validation is engineering agreement, not clinical validation
- Phase 4C.4 replays structured classifier outputs through the full
  evidence→policy chain without model inference; candidate remains shadow-only
- Phase 4C.5 adds authority migration comparison controls; default remains
  legacy; candidate authority is not activated
- No multimodal / facial / OCR personal-attribution path (by design)
- Ordinary unit tests mock the classifier; no live DeBERTa in normal pytest

## Calibration plan

1. Collect in-domain labeled examples (student social / campus-like text)
2. Measure relevance / target accuracy and signal precision-recall
3. Calibrate `WELLBEING_SIGNAL_POC_THRESHOLD` on a held-out validation set
4. Decide whether to keep zero-shot DeBERTa, fine-tune, or swap models
   while preserving schemas
5. Only then wire into the wellbeing gate / final assessment
