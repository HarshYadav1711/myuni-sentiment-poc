# Benchmark datasets for MyUni Multimodal Sentiment POC

This repository **does not bundle or redistribute** TweetEval, MVSA, CMU-MOSI, or CMU-MOSEI media/labels. Obtain data from the official sources below and respect each dataset’s license and terms.

Evaluation code lives under `evaluation/` and expects **prepared local JSONL indexes** (except optional TweetEval Hugging Face loading).

---

## TweetEval (text sentiment)

| Field | Value |
| --- | --- |
| Task used here | Sentiment (3-way) |
| Official GitHub | https://github.com/cardiffnlp/tweeteval |
| Hugging Face dataset | https://huggingface.co/datasets/cardiffnlp/tweet_eval (config: `sentiment`) |
| Paper | Barbieri et al., TweetEval (Findings of EMNLP 2020), https://arxiv.org/abs/2010.12421 |
| Labels (HF card) | `0` negative, `1` neutral, `2` positive |

### Acquisition

**Option A — Hugging Face (optional, network):**

```powershell
pip install datasets
python -m evaluation.run text --tweeteval-hf --split test --limit 100 --stub
# Real model (downloads weights):
python -m evaluation.run text --tweeteval-hf --split test --limit 50 --out outputs/eval_tweeteval
```

**Option B — Local JSONL** (one object per line):

```json
{"id":"t1","text":"I love this","label":"positive"}
```

`label` may be `0|1|2` or `negative|neutral|positive`.

Preparation helper (writes a tiny template only — does not download the corpus):

```powershell
python evaluation/text/prepare_tweeteval_template.py
```

---

## MVSA (image + text)

| Field | Value |
| --- | --- |
| Official project page | https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/ |
| Variants | MVSA-single, MVSA-multiple (see project page for current download links) |
| Citation | Niu et al., “Sentiment Analysis on Multi-view Social Data”, MMM 2016 |

### Acquisition (manual)

1. Visit the official MCRLab MVSA page and download **MVSA-single** and/or **MVSA-multiple** via the links published there (historically OneDrive / BaiduYun — confirm on the live page).
2. Review the license/terms provided with the release.
3. Build a local JSONL index (this repo does not ship images):

```json
{"id":"1","text":"campus fest","image_path":"images/1.jpg","label":"positive","text_label":"positive","image_label":"neutral"}
```

Template generator:

```powershell
python evaluation/image/prepare_mvsa_template.py --out data/eval/mvsa_index.jsonl
```

Then run:

```powershell
python -m evaluation.run image --data data/eval/mvsa_index.jsonl --limit 20 --stub
```

---

## CMU-MOSI (video multimodal)

| Field | Value |
| --- | --- |
| Official tooling | https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK |
| MOSI notes in SDK | `mmsdk/.../standard_datasets/CMU_MOSI/README.md` |
| Typical continuous label range (SDK README) | approximately **[-3, +3]** |
| Paper | Zadeh et al., MOSI corpus (arXiv:1606.06259) |

SDK README also documents **binary** and **7-class** conventions used in papers. This POC does **not** silently convert labels.

### Explicit POC 3-way mapping (documented)

For optional 3-way classification metrics only:

- `score < 0` → `negative`
- `score > 0` → `positive`
- `score == 0` → `neutral`

Continuous metrics are also reported: **MAE** and **Pearson correlation**, comparing `gold_score / 3` to the model’s `pred_score` in `[-1, +1]`.

### Acquisition

1. Follow CMU MultimodalSDK documentation to obtain CMU-MOSI features/labels/raw media as permitted.
2. Export a local JSONL index (paths to clips you are allowed to use):

```json
{"id":"seg1","video_path":"clips/seg1.mp4","score":2.1,"text":"this movie was great"}
```

Template:

```powershell
python evaluation/video/prepare_mosi_template.py --out data/eval/mosi_index.jsonl
```

```powershell
python -m evaluation.run video --data data/eval/mosi_index.jsonl --limit 10 --stub
```

---

## CMU-MOSEI (future larger benchmark)

CMU-MOSEI is a larger multimodal sentiment/emotion corpus available via the same CMU MultimodalSDK ecosystem:

- SDK: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK

It is **documented here as a future benchmark** and is **not required** for this MVP evaluation path unless you extend adapters similarly to MOSI.

---

## Outputs

Each run writes:

- `metrics.json` — machine-readable metrics + notes
- `predictions.csv` — per-example gold/pred

Console prints a short human-readable summary.

Use `--limit N` to evaluate a small subset first.
