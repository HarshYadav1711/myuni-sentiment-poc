## Controlled video validation procedure

Purpose: validate deterministic temporal context and advisory temporal reasoning on a user-supplied 15–20 second video with known spoken progression.

### Suggested spoken progression

- 0–5s: `Today has been pretty normal.`
- 5–10s: `I've got three assignments due tomorrow.`
- 10–15s: `I'm getting really exhausted trying to finish everything.`
- 15–20s: `It's honestly becoming pretty overwhelming.`

### Recording guidance

- Keep one speaker and steady framing.
- Use clear speech and minimal background noise.
- Avoid jump cuts.
- Keep lighting stable so visual evidence changes do not dominate unintentionally.
- If adding on-screen text, document the exact text and when it appears.

### Run configuration

1. Use the existing video pipeline unchanged.
2. Keep deterministic temporal windowing at `5.0` seconds.
3. Run once with reasoner disabled to inspect only deterministic context.
4. Run again with the reasoner enabled.
5. For later comparison, record whether decoding used:
   - deterministic-like profile: `temperature=0.0`, `top_p=1.0`, `top_k=0`
   - evaluation profile: `temperature=0.7`, `top_p=0.8`, `top_k=20`

### What to inspect

#### A. Whisper transcript and timestamps

- Compare transcript text to the spoken script.
- Confirm segment `start` / `end` ordering.
- Note whether segment boundaries roughly track the four 5-second phases.

#### B. RoBERTa speech sentiment

- Review transcript sentiment evidence overall.
- Review any per-window speech probability aggregates.
- Confirm whether later windows become more negative than earlier ones.

#### C. Visual evidence

- Review sampled frame timestamps.
- Inspect per-window visual probabilities.
- Note whether visuals stay stable, become more negative, or remain weak/sparse.

#### D. Deterministic temporal windows

For each window, record:

- window id (`window-0`, `window-1`, ...)
- start / end
- available modalities
- dominant sentiment
- visual probabilities
- speech probabilities
- OCR probabilities
- any conflict markers

#### E. Deterministic feature checks

Verify:

- normalized trajectory label
- negative persistence
- longest negative run
- strongest negative window
- sudden negative change
- cross-modal agreement
- cross-modal conflicts
- evidence coverage

### Expected pattern for the suggested script

This is a validation expectation, not a hard assertion:

- earlier windows should be neutral or mildly negative
- later windows should trend more negative
- trajectory may become `increasing_negative`
- strongest negative window is likely the last or second-last window
- persistence should reflect whether negativity spans multiple usable windows

### LLM reasoning checks

Treat the reasoner as advisory only.

Check that it:

- explains the deterministic trajectory instead of redefining it
- references only supplied evidence ids
- describes conflicts only when deterministic conflict exists
- avoids clinical / wellbeing claims
- states uncertainty when evidence is sparse
- keeps transition timestamps inside supplied windows

### Report template

1. Video metadata: duration, audio present, sampled frame count
2. Whisper segments: text + timestamps
3. Deterministic windows summary
4. Deterministic features summary
5. Reasoner output summary
6. Evidence ids cited by reasoner
7. Any contradictions between reasoner text and deterministic facts
8. Timing:
   - model load
   - prompt construction
   - generation
   - parse / validation
   - total reasoner time
9. Final note: whether the clip is suitable for later wellbeing-layer evaluation
