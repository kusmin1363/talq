# Datasets

Every corpus is public, but most need a separate download or a licence
acknowledgement. Splits follow the s3prl SUPERB recipes, so the numbers stay
comparable with the published SUPERB table.

Point the code at your copies with one variable:

```bash
export TALQ_DATA_ROOT=/path/to/corpora
```

## Expected layout

`talq/paths.py` is the single source of truth; this is what it resolves to.

```
$TALQ_DATA_ROOT/
├── librispeech/LibriSpeech/{train-clean-100,test-clean,dev-clean}/   PR, ASR
├── superb/
│   ├── speech_commands/                                             KS
│   └── fluent_speech_commands_dataset/                              IC
├── voxceleb1/{dev,test}/wav/                                        ASV
├── iemocap.jsonl                                                    ER (manifest)
├── librispeech-lexicon.txt                                          PR (task-head lexicon)
└── emilia/Emilia/EN/                                                calibration audio
```

| Task | Corpus | Split | Metric |
|---|---|---|---|
| PR  | LibriSpeech | train-clean-100 / test-clean (all 2,608 utterances) | PER |
| ASR | LibriSpeech | train-clean-100 / test-clean (all 2,608) | WER |
| KS  | Speech Commands v0.01 | official test tarball, all 3,081 | error rate |
| IC  | Fluent Speech Commands | all 3,793; an utterance counts only if all three slots match | error rate |
| ER  | IEMOCAP | 4-way, five session folds, held-out session per fold | error rate |
| ASV | VoxCeleb1 | dev for training, all 37,720 official trial pairs | EER |

Speech Commands ships its test set as a **separate tarball**. Do not hold out part
of train instead — that is the classic way to get an inflated KS number.

## Two different LibriSpeech lexicons

They are not interchangeable and the code keeps them apart on purpose.

- `librispeech-lexicon.txt` under `$TALQ_DATA_ROOT` is what the task heads were
  **trained** against. It strips stress, giving 39 phones, and drops utterances it
  cannot cover.
- The official s3prl lexicon (below) keeps stress, giving 71 phones and 0% OOV on
  train-clean-100. It is what evaluation scores against.

Pointing the heads at the official lexicon changes the symbol inventory and every
PR number.

## s3prl

PR and ASR score against s3prl's published CTC recipe, and ASV needs its dev-split
metadata, so a checkout is required rather than a copy of the files:

```bash
git clone https://github.com/s3prl/s3prl
export TALQ_S3PRL=/path/to/s3prl/s3prl/downstream
```

Used from it: `ctc/lexicon/librispeech-lexicon-{200k,allothers}-g2p.txt`,
`ctc/vocab/phoneme.txt`, and `sv_voxceleb1/dev_meta_data/dev_meta_data.txt`.

## Calibration audio

The PTQ candidates are built from Emilia (EN). `talq.data_prep.make_task_calib`
draws the per-task calibration sets from each task's own training split instead;
that is what the paper's 1,024-second candidates use.

## Pretrained backbones

Pulled from the Hugging Face hub at run time; no weights ship with this repo.
`facebook/wav2vec2-base`, `facebook/hubert-base-ls960`, `microsoft/wavlm-base`,
`facebook/hubert-large-ll60k`, `microsoft/wavlm-large`.
