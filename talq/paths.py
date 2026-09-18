"""Filesystem layout, resolved once, in one place.

Nothing in this repository hard-codes an absolute path. Every root below is an
environment variable with a repository-relative default, so a fresh checkout runs
unmodified once the data is either placed under the repo or pointed at:

    export TALQ_DATA_ROOT=/mnt/corpora              # datasets
    export TALQ_CKPT_ROOT=/mnt/talq/checkpoints     # task heads
    export TALQ_QUANT_ROOT=/mnt/talq/quants         # quantized candidates
    export TALQ_S3PRL=/opt/s3prl/s3prl/downstream   # an s3prl checkout

Read roots (data, checkpoints, candidates, s3prl) are never written to. Write
roots (results, logs) are created on import.

The four read roots differ in how you obtain them:

    DATA_ROOT    public corpora, downloaded once           docs/DATA.md
    S3PRL        a checkout of s3prl, for the CTC lexicon  docs/DATA.md
                 and the ASV dev split metadata
    CKPT_ROOT    trained task heads, published separately  docs/CHECKPOINTS.md
    QUANT_ROOT   quantized candidates, ~790 GB, generated  docs/REPRODUCE.md
                 locally by talq.quant.precompute

The subdirectory names below are not free choices -- they are the layout the
released checkpoints and the shipped results were produced against.
"""
import os
from pathlib import Path

__all__ = [
    "PACKAGE_ROOT", "REPO_ROOT",
    "DATA_ROOT", "CKPT_ROOT", "QUANT_ROOT", "CALIB_ROOT", "RESULTS_ROOT", "LOGS_ROOT", "TB_ROOT",
    "S3PRL", "S3PRL_CTC", "LEXICONS", "PHONEME_VOCAB", "ASV_DEV_META",
    "LIBRISPEECH", "LIBRISPEECH_LEXICON", "SUPERB_ROOT", "SPEECH_COMMANDS", "FLUENT_SPEECH",
    "VOXCELEB1", "IEMOCAP_MANIFEST", "EMILIA_EN",
    "QUANT_GPTQ", "QUANT_AWQ",
    "PROBES", "PROBES_SUPERB",
]


def _root(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, default)).expanduser()


PACKAGE_ROOT = Path(__file__).resolve().parent      # <repo>/talq
REPO_ROOT = PACKAGE_ROOT.parent                     # <repo>

# ---- roots -----------------------------------------------------------------
DATA_ROOT = _root("TALQ_DATA_ROOT", REPO_ROOT / "data")
CKPT_ROOT = _root("TALQ_CKPT_ROOT", REPO_ROOT / "checkpoints")
QUANT_ROOT = _root("TALQ_QUANT_ROOT", REPO_ROOT / "quants")
CALIB_ROOT = _root("TALQ_CALIB_ROOT", REPO_ROOT / "calib")
RESULTS_ROOT = _root("TALQ_RESULTS_ROOT", REPO_ROOT / "results")
LOGS_ROOT = _root("TALQ_LOGS_ROOT", REPO_ROOT / "logs")
TB_ROOT = LOGS_ROOT / "tbruns"

# ---- s3prl -----------------------------------------------------------------
# PR and ASR score against s3prl's published CTC recipe, so the lexicon and the
# phoneme vocabulary must come from an s3prl checkout rather than a copy: the
# published SUPERB numbers are defined by those exact files. ASV needs the dev
# split metadata, which lives in a sibling recipe directory.
S3PRL = _root("TALQ_S3PRL", REPO_ROOT / "third_party" / "s3prl" / "s3prl" / "downstream")
S3PRL_CTC = S3PRL / "ctc"
LEXICONS = [S3PRL_CTC / "lexicon" / "librispeech-lexicon-200k-g2p.txt",
            S3PRL_CTC / "lexicon" / "librispeech-lexicon-allothers-g2p.txt"]
PHONEME_VOCAB = S3PRL_CTC / "vocab" / "phoneme.txt"
ASV_DEV_META = S3PRL / "sv_voxceleb1" / "dev_meta_data" / "dev_meta_data.txt"

# ---- corpora ---------------------------------------------------------------
# Only Speech Commands and Fluent Speech Commands sit under a shared superb/
# directory; the rest are top-level. This mirrors how the corpora unpack.
LIBRISPEECH = DATA_ROOT / "librispeech" / "LibriSpeech"
SUPERB_ROOT = DATA_ROOT / "superb"
SPEECH_COMMANDS = SUPERB_ROOT / "speech_commands"
FLUENT_SPEECH = SUPERB_ROOT / "fluent_speech_commands_dataset"
VOXCELEB1 = DATA_ROOT / "voxceleb1"
IEMOCAP_MANIFEST = DATA_ROOT / "iemocap.jsonl"
EMILIA_EN = DATA_ROOT / "emilia" / "Emilia" / "EN"

# Only talq.eval.probe_train's default reads this one; it strips stress, giving 39
# phones instead of 71, and drops the utterances it cannot cover. The published PR
# and ASR heads were trained on the official s3prl lexicon in LEXICONS (their itos
# is the 71-symbol vocab), and every evaluation path calls load_lexicon_official.
# Keep the two apart -- the symbol inventory sets every PR number.
LIBRISPEECH_LEXICON = DATA_ROOT / "librispeech-lexicon.txt"

# ---- quantized candidates --------------------------------------------------
# QUANT_ROOT is the CONTAINER: each candidate set is a directory under it, and
# --quant-dir names one relative to this root (e.g. "quants_tc1024_pr", or
# "quants_1024/GPTQ/quants_tc1024_pr" for the per-task 1,024 s candidates the
# paper uses). The two below are the generic single-calibration sets.
QUANT_GPTQ = QUANT_ROOT / "quants"
QUANT_AWQ = QUANT_ROOT / "quants_awq"

# ---- task heads ------------------------------------------------------------
# PROBES holds one head per (backbone, task); PROBES_SUPERB holds the heads whose
# SUPERB recipe splits the data further -- ER into five session folds, and SID.
# ER must be read from PROBES_SUPERB: the single pooled ER head in PROBES was
# trained across sessions and leaks the held-out session of every fold.
PROBES = CKPT_ROOT / "probes"
PROBES_SUPERB = CKPT_ROOT / "probes_superb"


for _d in (RESULTS_ROOT, LOGS_ROOT):
    _d.mkdir(parents=True, exist_ok=True)
