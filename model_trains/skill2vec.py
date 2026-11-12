#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train posting-level Skill2Vec from the filtered train split.

One sentence is the set of unique ``true_skill`` labels belonging to one
``(file_path, row_idx)`` posting in ``all_train_test/train``. Skill order is
reshuffled deterministically for every training epoch. The test split is never read.

The train files are scanned once with up to eight workers and merged into a
disk-backed corpus. Vocabulary construction and every Word2Vec epoch stream
that cache instead of keeping millions of posting sentences in memory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import random
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator, TextIO

import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

BASE_DATA_DIR = Path(
    os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data3/hyunjincho/LEM")
)
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(BASE_DATA_DIR / "all_train_test")))
TRAIN_ROOT = Path(os.environ.get("TRAIN_ROOT", str(DATA_ROOT / "train")))
VOCAB_PATH = Path(
    os.environ.get("VOCAB_PATH", str(BASE_DATA_DIR / "all_filterd" / "skill2idx.json"))
)
OUT_DIR = Path(os.environ.get("SKILL2VEC_DIR", str(BASE_DATA_DIR / "skill2vec_new")))

PATTERN = "preprocessed_*.csv.gz"
PROGRESS_EVERY = 100
JOINER = "_"
VECTOR_SIZE = 300
WINDOW = 50
MIN_COUNT = 1
SG = 1
NEGATIVE = 10
EPOCHS = 5
MAX_WORKERS = 8
WORKERS = min(MAX_WORKERS, os.cpu_count() or 1)
SEED = 42
RESUME = True
EXPORT_TOPN: int | None = None
REBUILD_CORPUS = False

MODEL_NAME = (
    f"skill2vec_norm_sg{SG}_d{VECTOR_SIZE}_win{WINDOW}_neg{NEGATIVE}_ep{EPOCHS}"
)
MODEL_PATH = OUT_DIR / f"{MODEL_NAME}.model"
KV_PATH = OUT_DIR / f"{MODEL_NAME}.kv"
CORPUS_PATH = OUT_DIR / f"{MODEL_NAME}.corpus.txt"
CORPUS_META_PATH = OUT_DIR / f"{MODEL_NAME}.corpus.txt.json"
TOKEN_MAP_PATH = OUT_DIR / f"{MODEL_NAME}.skill_tokens.json"
TRAINING_META_PATH = OUT_DIR / f"{MODEL_NAME}.training.json"

CACHE_SCHEMA_VERSION = 2
NORMALIZATION_VERSION = "preprocess_plang_lower_strip_v1"
REQUIRED_COLUMNS = ("row_idx", "true_skill", "file_path")
_PLANG_TAIL = re.compile(r"\s*\(programming language\)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class CorpusTask:
    sequence: int
    input_path: str
    shard_dir: str


@dataclass(frozen=True)
class CorpusResult:
    sequence: int
    shard_path: str
    input_rows: int
    postings: int
    sentences: int
    tokens: int
    duplicate_skill_rows: int
    single_skill_postings: int


_WORKER_SKILL_TO_TOKEN: dict[str, str] | None = None


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Skill2Vec from unique true_skill labels grouped by train " "posting."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Split root containing train/ and test/ (test is never read).",
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        help="Train split root (default: <data-root>/train).",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=VOCAB_PATH,
        help="Final filtered skill2idx.json.",
    )
    parser.add_argument("--pattern", default=PATTERN)
    parser.add_argument("--progress-every", type=int, default=PROGRESS_EVERY)
    parser.add_argument("--joiner", default=JOINER)
    parser.add_argument("--vector-size", type=int, default=VECTOR_SIZE)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--min-count", type=int, default=MIN_COUNT)
    parser.add_argument("--sg", type=int, choices=(0, 1), default=SG)
    parser.add_argument("--negative", type=int, default=NEGATIVE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help="Corpus processes/Gensim workers (1-8; phases do not overlap).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--model-name",
        default="",
        help="Empty preserves the historical compatible model filename.",
    )
    parser.add_argument("--resume", type=parse_bool, default=RESUME)
    parser.add_argument(
        "--export-topn",
        type=int,
        default=0,
        help="0 exports all vectors.",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        help="Default: <output-dir>/<model-name>.corpus.txt.",
    )
    parser.add_argument(
        "--rebuild-corpus",
        type=parse_bool,
        default=REBUILD_CORPUS,
        help="Force a rescan even when a matching corpus cache exists.",
    )
    return parser.parse_args()


def configure(args: argparse.Namespace) -> None:
    global DATA_ROOT, TRAIN_ROOT, VOCAB_PATH, OUT_DIR, PATTERN
    global PROGRESS_EVERY, JOINER, VECTOR_SIZE, WINDOW, MIN_COUNT
    global SG, NEGATIVE, EPOCHS, WORKERS, SEED, RESUME, EXPORT_TOPN
    global REBUILD_CORPUS, MODEL_NAME, MODEL_PATH, KV_PATH, CORPUS_PATH
    global CORPUS_META_PATH, TOKEN_MAP_PATH, TRAINING_META_PATH

    positive = (
        args.progress_every,
        args.vector_size,
        args.window,
        args.epochs,
        args.workers,
    )
    if any(value < 1 for value in positive):
        raise ValueError(
            "progress interval, vector size, window, epochs, and workers "
            "must be >= 1"
        )
    if args.workers > MAX_WORKERS:
        raise ValueError(f"--workers must be <= {MAX_WORKERS}")
    if args.min_count < 0 or args.negative < 0:
        raise ValueError("--min-count and --negative must be >= 0")
    if args.export_topn < 0 or args.seed < 0:
        raise ValueError("--export-topn and --seed must be >= 0")
    if not args.pattern:
        raise ValueError("--pattern must not be empty")
    if not args.joiner or any(char.isspace() for char in args.joiner):
        raise ValueError("--joiner must be non-empty and contain no whitespace")

    DATA_ROOT = args.data_root.expanduser().resolve()
    TRAIN_ROOT = (
        args.train_root.expanduser().resolve()
        if args.train_root is not None
        else (DATA_ROOT / "train").resolve()
    )
    VOCAB_PATH = args.vocab_path.expanduser().resolve()
    OUT_DIR = args.output_dir.expanduser().resolve()
    PATTERN = args.pattern
    PROGRESS_EVERY = args.progress_every
    JOINER = args.joiner
    VECTOR_SIZE = args.vector_size
    WINDOW = args.window
    MIN_COUNT = args.min_count
    SG = args.sg
    NEGATIVE = args.negative
    EPOCHS = args.epochs
    WORKERS = args.workers
    SEED = args.seed
    RESUME = args.resume
    EXPORT_TOPN = None if args.export_topn == 0 else args.export_topn
    REBUILD_CORPUS = args.rebuild_corpus

    MODEL_NAME = args.model_name or (
        f"skill2vec_norm_sg{SG}_d{VECTOR_SIZE}_win{WINDOW}_" f"neg{NEGATIVE}_ep{EPOCHS}"
    )
    MODEL_PATH = OUT_DIR / f"{MODEL_NAME}.model"
    KV_PATH = OUT_DIR / f"{MODEL_NAME}.kv"
    CORPUS_PATH = (
        args.corpus_path.expanduser().resolve()
        if args.corpus_path is not None
        else OUT_DIR / f"{MODEL_NAME}.corpus.txt"
    )
    CORPUS_META_PATH = CORPUS_PATH.with_suffix(CORPUS_PATH.suffix + ".json")
    TOKEN_MAP_PATH = OUT_DIR / f"{MODEL_NAME}.skill_tokens.json"
    TRAINING_META_PATH = OUT_DIR / f"{MODEL_NAME}.training.json"


def normalize_dataset_skill(skill: str) -> str | None:
    """Apply preprocess.py's exact vocabulary-membership normalization."""
    if not isinstance(skill, str):
        return None
    normalized = _PLANG_TAIL.sub("", skill.strip()).strip().lower()
    return normalized or None


def skill_to_token(skill: str) -> str:
    """Encode whitespace only after filtered-vocabulary membership succeeds."""
    token = JOINER.join(skill.split())
    if not token or any(char.isspace() for char in token):
        raise ValueError(f"invalid single-token encoding for {skill!r}")
    return token


def load_filtered_vocab(path: Path) -> tuple[dict[str, str], dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"filtered vocabulary not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} must contain a non-empty skill->index object")

    skill_to_index: dict[str, int] = {}
    skill_to_token_map: dict[str, str] = {}
    token_to_skill: dict[str, str] = {}
    indices: set[int] = set()
    for raw_skill, raw_index in payload.items():
        if not isinstance(raw_skill, str) or not isinstance(raw_index, int):
            raise ValueError(f"invalid skill2idx entry: {raw_skill!r}: {raw_index!r}")
        skill = normalize_dataset_skill(raw_skill)
        if skill != raw_skill:
            raise ValueError(f"vocabulary key is not normalized: {raw_skill!r}")
        if raw_index < 0 or raw_index in indices:
            raise ValueError(f"duplicate/negative vocabulary index: {raw_index}")
        token = skill_to_token(skill)
        previous = token_to_skill.get(token)
        if previous is not None and previous != skill:
            raise ValueError(
                f"Word2Vec token collision: {previous!r}, {skill!r} -> {token!r}"
            )
        indices.add(raw_index)
        skill_to_index[skill] = raw_index
        skill_to_token_map[skill] = token
        token_to_skill[token] = skill
    if indices != set(range(len(skill_to_index))):
        raise ValueError("skill2idx indices must be contiguous from zero")
    print(f"loaded filtered vocabulary: {len(skill_to_token_map):,} skills")
    return skill_to_token_map, skill_to_index


def open_processed_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="\n")
    return path.open("r", encoding="utf-8", newline="\n")


def normalized_csv_lines(handle: TextIO) -> Iterator[str]:
    """Keep legacy bare CR characters from becoming CSV record breaks."""
    for line in handle:
        yield line.replace("\r\n", "\n").replace("\r", " ")


def parse_row_idx(value: str, path: Path) -> int:
    try:
        row_idx = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} has invalid row_idx={value!r}") from exc
    if row_idx < 0:
        raise ValueError(f"{path} has negative row_idx={row_idx}")
    return row_idx


def stable_epoch_seed(seed: int, epoch: int, line_number: int) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"LEM-s2v-v2")
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(epoch).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(line_number).encode("ascii"))
    return int.from_bytes(digest.digest(), "big")


def init_corpus_worker(skill_to_token_map: dict[str, str]) -> None:
    global _WORKER_SKILL_TO_TOKEN
    _WORKER_SKILL_TO_TOKEN = skill_to_token_map


def build_corpus_shard(task: CorpusTask) -> CorpusResult:
    if _WORKER_SKILL_TO_TOKEN is None:
        raise RuntimeError("corpus worker was not initialized")
    input_path = Path(task.input_path)
    posting_skills: dict[tuple[str, int], set[str]] = {}
    input_rows = 0
    duplicate_skill_rows = 0

    with open_processed_csv(input_path) as input_handle:
        reader = csv.reader(normalized_csv_lines(input_handle))
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{input_path} is empty and has no header") from exc
        missing = [column for column in REQUIRED_COLUMNS if column not in header]
        if missing:
            raise ValueError(f"{input_path} is missing columns: {', '.join(missing)}")
        positions = {column: header.index(column) for column in REQUIRED_COLUMNS}
        max_position = max(positions.values())
        for row in reader:
            if len(row) <= max_position:
                raise ValueError(f"{input_path} contains a short CSV row")
            input_rows += 1
            source_path = row[positions["file_path"]]
            if not source_path:
                raise ValueError(f"{input_path} contains an empty file_path")
            row_idx = parse_row_idx(row[positions["row_idx"]], input_path)
            skill = normalize_dataset_skill(row[positions["true_skill"]])
            if skill is None:
                raise ValueError(f"{input_path} contains an empty true_skill")
            token = _WORKER_SKILL_TO_TOKEN.get(skill)
            if token is None:
                raise ValueError(
                    f"{input_path} true_skill is absent from filtered vocab: "
                    f"{skill!r}"
                )
            skills = posting_skills.setdefault((source_path, row_idx), set())
            before = len(skills)
            skills.add(token)
            if len(skills) == before:
                duplicate_skill_rows += 1

    shard_path = Path(task.shard_dir) / f"part-{task.sequence:08d}.txt"
    sentences = 0
    tokens_written = 0
    single_skill_postings = 0
    with shard_path.open("w", encoding="utf-8", newline="\n") as shard_handle:
        for _, skill_tokens in sorted(posting_skills.items()):
            if len(skill_tokens) < 2:
                single_skill_postings += 1
                continue
            sentence = sorted(skill_tokens)
            shard_handle.write(" ".join(sentence))
            shard_handle.write("\n")
            sentences += 1
            tokens_written += len(sentence)

    return CorpusResult(
        sequence=task.sequence,
        shard_path=str(shard_path),
        input_rows=input_rows,
        postings=len(posting_skills),
        sentences=sentences,
        tokens=tokens_written,
        duplicate_skill_rows=duplicate_skill_rows,
        single_skill_postings=single_skill_postings,
    )


def discover_train_files(root: Path, pattern: str) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"train split root not found: {root}")
    files = sorted(path for path in root.rglob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} under {root}")
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_signature(train_files: list[Path]) -> str:
    digest = hashlib.sha256()
    settings = {
        "schema": CACHE_SCHEMA_VERSION,
        "normalization": NORMALIZATION_VERSION,
        "train_root": str(TRAIN_ROOT),
        "pattern": PATTERN,
        "joiner": JOINER,
        "seed": SEED,
        "vocab_path": str(VOCAB_PATH),
        "vocab_sha256": sha256_file(VOCAB_PATH),
    }
    digest.update(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for path in train_files:
        stat = path.stat()
        digest.update(str(path.relative_to(TRAIN_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_matching_corpus_metadata(signature: str) -> dict[str, object] | None:
    if REBUILD_CORPUS or not CORPUS_PATH.is_file() or not CORPUS_META_PATH.is_file():
        return None
    try:
        with CORPUS_META_PATH.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        return None
    if metadata.get("corpus_signature") != signature:
        return None
    if int(metadata.get("sentences", 0)) < 1:
        return None
    return metadata


def build_disk_corpus(
    train_files: list[Path],
    skill_to_token_map: dict[str, str],
    signature: str,
) -> dict[str, object]:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    totals = {
        "input_rows": 0,
        "postings": 0,
        "sentences": 0,
        "tokens": 0,
        "duplicate_skill_rows": 0,
        "single_skill_postings": 0,
    }
    temporary_corpus = CORPUS_PATH.with_name(f".{CORPUS_PATH.name}.tmp.{os.getpid()}")
    try:
        with tempfile.TemporaryDirectory(
            prefix="skill2vec-shards-", dir=str(CORPUS_PATH.parent)
        ) as shard_dir, temporary_corpus.open("wb") as corpus_handle:
            tasks = [
                CorpusTask(sequence, str(path), shard_dir)
                for sequence, path in enumerate(train_files)
            ]
            with Pool(
                processes=WORKERS,
                initializer=init_corpus_worker,
                initargs=(skill_to_token_map,),
                maxtasksperchild=200,
            ) as pool:
                results = pool.imap(build_corpus_shard, tasks, chunksize=1)
                for completed, result in enumerate(results, start=1):
                    shard_path = Path(result.shard_path)
                    with shard_path.open("rb") as shard_handle:
                        shutil.copyfileobj(shard_handle, corpus_handle, 1024 * 1024)
                    shard_path.unlink()
                    totals["input_rows"] += result.input_rows
                    totals["postings"] += result.postings
                    totals["sentences"] += result.sentences
                    totals["tokens"] += result.tokens
                    totals["duplicate_skill_rows"] += result.duplicate_skill_rows
                    totals["single_skill_postings"] += result.single_skill_postings
                    if completed % PROGRESS_EVERY == 0 or completed == len(tasks):
                        print(
                            f"built corpus files {completed:,}/{len(tasks):,}; "
                            f"postings={totals['postings']:,}; "
                            f"sentences={totals['sentences']:,}"
                        )
        if totals["sentences"] < 1:
            raise RuntimeError("no train posting has at least two unique skills")
        os.replace(temporary_corpus, CORPUS_PATH)
    finally:
        if temporary_corpus.exists():
            temporary_corpus.unlink()

    metadata: dict[str, object] = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "corpus_signature": signature,
        "corpus_path": str(CORPUS_PATH),
        "train_root": str(TRAIN_ROOT),
        "test_split_used": False,
        "sentence_definition": (
            "unique true_skill grouped by (file_path,row_idx), minimum 2"
        ),
        "pattern": PATTERN,
        "train_files": len(train_files),
        "vocab_path": str(VOCAB_PATH),
        "filtered_vocab_size": len(skill_to_token_map),
        "normalization_version": NORMALIZATION_VERSION,
        "posting_skill_order": "reshuffle_each_epoch_blake2b(seed,epoch,line)",
        "workers": WORKERS,
        **totals,
    }
    atomic_write_json(CORPUS_META_PATH, metadata)
    return metadata


class DiskSkillCorpus:
    """Re-iterable corpus that reshuffles every posting for each epoch."""

    def __init__(self, path: Path, seed: int):
        self.path = path
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("corpus epoch must be >= 0")
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[str]]:
        epoch = self.epoch
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                sentence = line.rstrip("\n").split()
                if len(sentence) < 2:
                    raise RuntimeError(
                        f"invalid cached sentence at {self.path}:{line_number}"
                    )
                random.Random(stable_epoch_seed(self.seed, epoch, line_number)).shuffle(
                    sentence
                )
                yield sentence


# ─── Disk-streamed Word2Vec training ──────────────────────────────
def model_config() -> dict[str, object]:
    return {
        "vector_size": VECTOR_SIZE,
        "window": WINDOW,
        "min_count": MIN_COUNT,
        "sg": SG,
        "negative": NEGATIVE,
        "seed": SEED,
        "joiner": JOINER,
    }


def load_resume_state(signature: str) -> tuple[Word2Vec, int] | None:
    if not RESUME or not MODEL_PATH.is_file():
        return None
    if not TRAINING_META_PATH.is_file():
        raise RuntimeError(
            f"cannot safely resume {MODEL_PATH}: missing {TRAINING_META_PATH}; "
            "run with --resume false"
        )
    with TRAINING_META_PATH.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("corpus_signature") != signature:
        raise RuntimeError(
            "existing model used another corpus; choose a new --model-name "
            "or run with --resume false"
        )
    if state.get("model_config") != model_config():
        raise RuntimeError(
            "existing model hyperparameters differ; choose a new --model-name "
            "or run with --resume false"
        )
    print(f"resuming model: {MODEL_PATH}")
    model = Word2Vec.load(str(MODEL_PATH))
    model.workers = WORKERS
    model.compute_loss = True
    return model, int(state.get("completed_epochs", 0))


def training_state(
    signature: str,
    corpus_metadata: dict[str, object],
    model: Word2Vec,
    completed_epochs: int,
) -> dict[str, object]:
    return {
        "model_path": str(MODEL_PATH),
        "keyed_vectors_path": str(KV_PATH),
        "corpus_path": str(CORPUS_PATH),
        "corpus_signature": signature,
        "model_config": model_config(),
        "workers": WORKERS,
        "epochs_requested_this_run": EPOCHS,
        "completed_epochs": completed_epochs,
        "trained_vocab_size": len(model.wv),
        "filtered_vocab_size": corpus_metadata["filtered_vocab_size"],
        "train_postings": corpus_metadata["postings"],
        "training_sentences": corpus_metadata["sentences"],
        "training_tokens_per_epoch": corpus_metadata["tokens"],
        "test_split_used": False,
        "software": {
            "python": platform.python_version(),
            "gensim": gensim.__version__,
        },
    }


def train_skill2vec(
    corpus: DiskSkillCorpus,
    corpus_metadata: dict[str, object],
    signature: str,
) -> Word2Vec:
    resumed = load_resume_state(signature)
    if resumed is None:
        model = Word2Vec(
            vector_size=VECTOR_SIZE,
            window=WINDOW,
            min_count=MIN_COUNT,
            sg=SG,
            negative=NEGATIVE,
            workers=WORKERS,
            seed=SEED,
            compute_loss=True,
        )
        print("building Word2Vec vocabulary from disk corpus")
        corpus.set_epoch(0)
        model.build_vocab(corpus)
        if len(model.wv) == 0:
            raise RuntimeError("Word2Vec vocabulary is empty")
        completed_before = 0
        print(f"trained vocabulary: {len(model.wv):,} skills")
    else:
        model, completed_before = resumed

    total_examples = int(corpus_metadata["sentences"])
    total_words = int(corpus_metadata["tokens"])
    for epoch in range(1, EPOCHS + 1):
        completed_epochs = completed_before + epoch
        corpus.set_epoch(completed_epochs)
        started = time.time()
        model.train(
            corpus,
            total_examples=total_examples,
            total_words=total_words,
            epochs=1,
            report_delay=30.0,
            compute_loss=True,
        )
        model.save(str(MODEL_PATH))
        model.wv.save(str(KV_PATH))
        atomic_write_json(
            TRAINING_META_PATH,
            training_state(signature, corpus_metadata, model, completed_epochs),
        )
        print(
            f"epoch {epoch}/{EPOCHS}: {time.time() - started:.1f}s; "
            f"loss={model.get_latest_training_loss():.2f}; saved {MODEL_PATH}"
        )
    return model


# ─── Export ─────────────────────────
def export_tsv(
    vectors: KeyedVectors,
    output_prefix: str,
    topn: int | None,
) -> None:
    keys = vectors.index_to_key if topn is None else vectors.index_to_key[:topn]
    vector_path = OUT_DIR / f"{output_prefix}.tsv"
    metadata_path = OUT_DIR / f"{output_prefix}.meta.tsv"
    with vector_path.open("w", encoding="utf-8") as vector_handle, metadata_path.open(
        "w", encoding="utf-8"
    ) as metadata_handle:
        for key in keys:
            vector_handle.write("\t".join(map(str, vectors[key].tolist())))
            vector_handle.write("\n")
            metadata_handle.write(key)
            metadata_handle.write("\n")
    print(f"exported vectors: {vector_path}")


# ─── Main ─────────────────────────
def main() -> int:
    configure(parse_args())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    skill_to_token_map, skill_to_index = load_filtered_vocab(VOCAB_PATH)
    atomic_write_json(
        TOKEN_MAP_PATH,
        {
            "vocab_path": str(VOCAB_PATH),
            "joiner": JOINER,
            "normalization_version": NORMALIZATION_VERSION,
            "skills": [
                {
                    "skill": skill,
                    "skill_index": skill_to_index[skill],
                    "word2vec_token": skill_to_token_map[skill],
                }
                for skill in sorted(skill_to_token_map, key=skill_to_index.get)
            ],
        },
    )

    train_files = discover_train_files(TRAIN_ROOT, PATTERN)
    signature = corpus_signature(train_files)
    print(f"train root:       {TRAIN_ROOT}")
    print(f"train files:      {len(train_files):,}")
    print("test split used:  no")
    print(f"workers:          {WORKERS}")
    print(f"corpus cache:     {CORPUS_PATH}")

    corpus_metadata = load_matching_corpus_metadata(signature)
    if corpus_metadata is None:
        corpus_metadata = build_disk_corpus(train_files, skill_to_token_map, signature)
        print(f"saved corpus cache: {CORPUS_PATH}")
    else:
        print(
            "reusing matching corpus cache: "
            f"{int(corpus_metadata['sentences']):,} sentences"
        )

    model = train_skill2vec(
        DiskSkillCorpus(CORPUS_PATH, SEED), corpus_metadata, signature
    )
    export_tsv(model.wv, f"{MODEL_NAME}_topall", EXPORT_TOPN)
    print(f"model:            {MODEL_PATH}")
    print(f"training summary: {TRAINING_META_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
