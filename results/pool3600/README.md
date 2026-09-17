# The 3,600-second allocation pool

The audio that bit allocation is decided on. One line per utterance, in the order
the pool is consumed.

This pool is **not** the quantization calibration set. The two axes are separate,
and both are held constant across the methods compared in the paper:

| | |
|---|---|
| quantization candidates | 1,024 s per-task calibration (`quants_tc1024_*`) |
| allocation decision | the 3,600 s pool listed here |

TALQ, the TAQ-KL baseline and Sens-DP all read this pool, which is what lets them
be described as seeing the same audio. An earlier round of the baseline decided
allocation from the 1,024 s calibration tensor instead and therefore saw 3.5×
less data than TALQ; `taq_grid --calib-pool 3600` is what closed that gap, and
these lists are what it selects.

## Why the lists are here at all

Nothing stores this pool. It is re-derived every run — `talq.eval.salience.train_pool`
indexes the training split, `random.Random(0)` shuffles it, and `int(3600 / MAX_S)`
items are taken from the front. That is deliberate: a stored waveform tensor would
freeze the crop length at one value, and the tasks need 8, 6 and 1 second contexts.

The side effect is that the pool is invisible to a reader — you would have to run
the code against the corpora to see what was in it. These lists remove that step.
Regenerate them with:

```bash
python -m talq.data_prep.dump_pool          # writes this directory
```

## Contents

| File | Corpus | `MAX_S` | Utterances | Training split |
|---|---|---|---|---|
| `pr.txt`, `asr.txt` | LibriSpeech | 8 s | 450 | `train-clean-100` |
| `ic.txt` | Fluent Speech Commands | 6 s | 600 | `train` |
| `ks.txt` | Speech Commands | 1 s | 3600 | `train` |
| `asv.txt` | VoxCeleb1 | 8 s | 450 | `dev` |
| `er_fold{1..5}.txt` | IEMOCAP | 6 s | 600 | the four sessions other than fold *n* |

Every list is exactly 3,600 s: the utterance count is `3600 / MAX_S`, and each is
read with `load_wav(max_s=MAX_S)`.

Paths are relative to the directory each corpus unpacks into — LibriSpeech and the
SUPERB corpora from `TALQ_DATA_ROOT`, IEMOCAP from `IEMOCAP_full_release/`.

## Two properties worth checking

- **`pr.txt` and `asr.txt` are identical.** Both index `train-clean-100` under the
  same seed, and the PR lexicon filter drops nothing at this scale (28,539
  utterances either way). PR and ASR therefore allocate against the same audio,
  and any difference between them comes from the task head, not the data.
- **`er_fold{n}.txt` contains no `Session{n}`.** ER is five-fold in SUPERB, so the
  session being scored must be absent from every training resource — the
  calibration, the task head and this pool. `grep -c Session1 er_fold1.txt`
  returns 0, and so on for all five.

## What is not here

The audio itself. All five corpora are public but separately licensed, so they are
obtained from their own distributors; see [`../../docs/DATA.md`](../../docs/DATA.md).
The crop position inside each utterance is drawn at run time, so these lists fix
*which* audio is seen, not the exact samples.
