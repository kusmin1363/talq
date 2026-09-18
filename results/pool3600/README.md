# The 3,600-second allocation pool

The audio bit allocation is decided on, one line per utterance, in the order the
pool is consumed. TALQ, the TAQ-KL baseline and Sens-DP all read this pool.

It is not stored anywhere else — `talq.eval.salience.train_pool` indexes the
training split, `random.Random(0)` shuffles it and `int(3600 / MAX_S)` items are
taken from the front, every run. Regenerate these lists with:

```bash
python -m talq.data_prep.dump_pool
```

| File | Corpus | `MAX_S` | Utterances |
|---|---|---|---|
| `pr.txt`, `asr.txt` | LibriSpeech `train-clean-100` | 8 s | 450 |
| `ic.txt` | Fluent Speech Commands `train` | 6 s | 600 |
| `ks.txt` | Speech Commands `train` | 1 s | 3600 |
| `asv.txt` | VoxCeleb1 `dev` | 8 s | 450 |
| `er_fold{1..5}.txt` | IEMOCAP, the four sessions other than fold *n* | 6 s | 600 |

Paths are relative to the directory each corpus unpacks into — the SUPERB corpora
and LibriSpeech from `TALQ_DATA_ROOT`, IEMOCAP from `IEMOCAP_full_release/`. The
audio itself is separately licensed; see [`../../docs/DATA.md`](../../docs/DATA.md).
