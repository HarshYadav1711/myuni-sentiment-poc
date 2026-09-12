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

`WELLBEING_SIGNAL_POC_THRESHOLD` (default `0.5`) controls which multi-label
signals are marked `selected=true`.

**This threshold is provisional and must be calibrated against an in-domain
validation set.** It is not clinically validated.

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

## Later connection to temporal video evidence

Video already produces speech transcripts, OCR text, and temporal windows.
A later milestone can feed **text evidence** (speech/OCR/captions) from
those windows into this classifier, then combine relevance/target/signals
with deterministic temporal features.

Visual appearance and face crops remain out of scope for wellbeing
classification.

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
- Signal threshold is uncalibrated
- Not yet connected to `compute_wellbeing_indicator()` or temporal fusion
- No multimodal / facial path (by design)
- Human semantic fixtures exist for later evaluation; ordinary unit tests
  mock the pipeline and do not require live model correctness

## Calibration plan

1. Collect in-domain labeled examples (student social / campus-like text)
2. Measure relevance / target accuracy and signal precision-recall
3. Calibrate `WELLBEING_SIGNAL_POC_THRESHOLD` on a held-out validation set
4. Decide whether to keep zero-shot DeBERTa, fine-tune, or swap models
   while preserving schemas
5. Only then wire into the wellbeing gate / final assessment
