"""Dataset indexers for the SUPERB tasks, in this project's (file, label) style.

Each function returns a list the trainer can `random.sample` from. Splits follow
s3prl's recipes so the numbers are comparable to the published SUPERB table:

  KS  speech_commands  12 classes = 10 words + _silence_ + _unknown_; the split
      is defined by validation_list.txt / testing_list.txt shipped in the tarball,
      and the official test set is a SEPARATE tarball (not the held-out part of
      train) -- mixing the two is the classic way to get an inflated KS number.
  IC  fluent_speech_commands  3 independent slots (action/object/location); SUPERB
      scores an utterance correct only when all three are right.
  ER  IEMOCAP 4-way (neu/hap+exc/ang/sad) with the s3prl session folds.
"""
import csv
import re
import glob
import os

from talq import paths

ROOT = paths.SUPERB_ROOT
KS_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
KS_CLASSES = KS_WORDS + ["_silence_", "_unknown_"]


def ks_index(split, root=paths.SPEECH_COMMANDS):
    """split: train | valid | test -> ([(wav, class_id)], sample_weights).

    Faithful to s3prl/downstream/speech_commands:
      * The split is NOT validation_list.txt. It is the original paper's
        speaker hash: sha1(speaker) % (2**27-1) -> <10%% valid, <20%% discarded,
        rest train. That keeps speakers disjoint; the .txt lists do not.
      * _silence_ has no directory in train. It is generated from the six
        _background_noise_ recordings, which the loader crops to a random 1 s
        window. Omitting it leaves class 11 untrained.
      * sample_weights = len(data)/count[class]. _unknown_ is ~64%% of the raw
        data, so without class-balanced sampling the probe just predicts it.
    """
    import hashlib
    c2i = {c: i for i, c in enumerate(KS_CLASSES)}
    MAX = 2 ** 27 - 1
    if split == "test":                       # the official test set is a separate tarball, already reduced to the 12 classes
        data = []
        for d in sorted(os.listdir(f"{root}/test")):
            if not os.path.isdir(f"{root}/test/{d}"):
                continue
            for f in sorted(glob.glob(f"{root}/test/{d}/*.wav")):
                data.append((f, c2i.get(d, c2i["_unknown_"])))
    else:
        tr = f"{root}/train"
        data = []
        for d in sorted(os.listdir(tr)):
            if not os.path.isdir(os.path.join(tr, d)) or d == "_background_noise_":
                continue
            lab = c2i.get(d, c2i["_unknown_"])
            for f in sorted(glob.glob(f"{tr}/{d}/*.wav")):
                spk = re.sub(r"_nohash_.*$", "", os.path.basename(f))
                pct = (int(hashlib.sha1(spk.encode()).hexdigest(), 16) % (MAX + 1)) * (100.0 / MAX)
                which = "valid" if pct < 10 else ("drop" if pct < 20 else "train")
                if which == split:
                    data.append((f, lab))
        data += [(f, c2i["_silence_"])
                 for f in sorted(glob.glob(f"{tr}/_background_noise_/*.wav"))]
    cnt = {}
    for _, l in data:
        cnt[l] = cnt.get(l, 0) + 1
    return data, [len(data) / cnt[l] for _, l in data]


def ic_index(split, root=paths.FLUENT_SPEECH):
    """split: train | valid | test. Returns [(wav, (act, obj, loc))] plus the
    three label vocabularies so the head knows its output sizes."""
    csvs = {"train": "train_data.csv", "valid": "valid_data.csv", "test": "test_data.csv"}
    vocab = [{}, {}, {}]
    for fn in csvs.values():                  # fix the label set over all three splits
        p = os.path.join(root, "data", fn)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            for i, k in enumerate(("action", "object", "location")):
                vocab[i].setdefault(r[k], len(vocab[i]))
    items = []
    for r in csv.DictReader(open(os.path.join(root, "data", csvs[split]))):
        items.append((os.path.join(root, r["path"]),
                      tuple(vocab[i][r[k]] for i, k in
                            enumerate(("action", "object", "location")))))
    return items, [len(v) for v in vocab]


def voxceleb_sid_index(dev_root):
    """SID/ASV share VoxCeleb1 dev. Layout is dev/wav/<spk>/<video>/*.wav -- the
    extra 'wav' level is easy to miss and silently yields an empty index.
    Returns ([(wav, spk_id)], speaker_list)."""
    base = os.path.join(dev_root, "wav") if os.path.isdir(os.path.join(dev_root, "wav")) else dev_root
    spks = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    s2i = {s: i for i, s in enumerate(spks)}
    items = [(f, s2i[s]) for s in spks
             for f in sorted(glob.glob(f"{base}/{s}/*/*.wav"))]
    assert items, f"VoxCeleb1 dev index is empty: {base}"
    return items, spks


VOX1 = paths.VOXCELEB1


def voxceleb_iden_index(split, root=VOX1):
    """Official SUPERB SID split. Column 1 of iden_split.txt is the split
    (1=train 2=dev 3=test).

    Two things differ from voxceleb_sid_index. They must not be mixed:
      * There are **1,251 speakers** (dev 1,211 + test 40). Our existing SID
        probe used dev only, so it had 1,211 outputs, and switching to the
        official split therefore forces a retrain.
      * The split is the official list, not random. The old code cut 8:2 with
        seed 0.

    The audio is spread over dev/wav and test/wav, so both are checked.
    Speaker ids are numbered by sorting over all of iden_split, so they are
    fixed independently of the split.
    Returns ([(wav, spk_id)], speaker_list).
    """
    rows = [ln.split() for ln in open(f"{root}/iden_split.txt")]
    spks = sorted({r[1].split("/")[0] for r in rows})
    s2i = {s: i for i, s in enumerate(spks)}
    want = str(split)
    items = []
    for sp, rel in rows:
        if sp != want:
            continue
        for sub in ("dev", "test"):
            p = f"{root}/{sub}/wav/{rel}"
            if os.path.exists(p):
                items.append((p, s2i[rel.split("/")[0]]))
                break
        else:
            raise FileNotFoundError(f"audio not found: {rel}")
    assert items, f"iden_split {split} index is empty"
    return items, spks


S3PRL = paths.S3PRL_CTC
OFFICIAL_LEXICONS = paths.LEXICONS
OFFICIAL_PHONEME_VOCAB = paths.PHONEME_VOCAB


def load_lexicon_official():
    """SUPERB PR's G2P, from downstream/ctc/libriphone.yaml.

    Two differences from talq.eval.probe_train.load_lexicon that both matter:
      * 214,716 entries across two files -> 0%% OOV on LibriSpeech. Ours drops
        7.42%% of train-clean-100, and the dropped set is biased (proper nouns,
        rare words), not a random sample.
      * Stress digits are KEPT (AA0/AA1/AA2 are distinct), giving the official
        70-symbol vocab. Ours strips them down to 39, which is a strictly easier
        task and makes our PER incomparable to the published SUPERB table.

    Returns (lex, phones) with the same shape as load_lexicon so callers are
    unchanged. `phones` comes from the official vocab file, not from the lexicon,
    so the symbol ORDER is fixed by s3prl rather than by our sorting.
    """
    lex = {}
    for path in OFFICIAL_LEXICONS:
        for line in open(path):
            parts = line.split()
            if parts:
                lex.setdefault(parts[0].upper(), parts[1:])
    phones = [p for p in open(OFFICIAL_PHONEME_VOCAB).read().split() if p]
    return lex, phones


# The dev split metadata lives in s3prl's sv_voxceleb1 recipe, a sibling of the
# ctc recipe the lexicon comes from -- hence talq.paths.S3PRL, not S3PRL_CTC.
ASV_DEV_META = paths.ASV_DEV_META


def asv_dev_trials(cap_wavs=1800, path=ASV_DEV_META):
    """An evaluation subset of s3prl's ASV dev trials (40 speakers, 20,000 pairs).

    The cost of EER evaluation during training is set not by the number of
    trials but by the **number of unique wavs** (one backbone forward per file).
    So we pick in 2 passes: first scan from the front to build a pool of
    cap_wavs wavs, then take **every trial whose two ends are both in the
    pool**. Cutting from the front only gives about one trial per wav, which
    makes the EER far too unstable.

    Being based on a fixed prefix, there is no randomness, and it is the same
    yardstick at every step and for every backbone.

    These 40 speakers have **zero intersection** with the 40 speakers of
    VoxCeleb1 test, so choosing here and reporting with veri_test2.txt keeps
    selection and reporting separate.
    """
    rows = [ln.split() for ln in open(path)]
    pool = set()
    for _, p1, p2 in rows:
        if len(pool | {p1, p2}) > cap_wavs:
            continue
        pool |= {p1, p2}
    return [(int(l), p1, p2) for l, p1, p2 in rows if p1 in pool and p2 in pool]
