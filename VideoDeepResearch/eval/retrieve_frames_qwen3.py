#!/usr/bin/env python3
"""
retrieve_frames_qwen3.py

For every video in frame_retriever_queries/, finds the most semantically
similar dense frames for each query using:

  1. A Nebius VLM (default: Qwen/Qwen3-VL-72B-Instruct) to generate a
     one-sentence visual caption for each uncached frame.
  2. Qwen3-Embedding (Nebius) to embed captions and queries.
  3. Cosine similarity to rank frames per query.

Captions and embeddings are persisted (see --cache-dir) so every API call
is made at most once — subsequent runs load from disk.

Usage
-----
  export NEBIUS_API_KEY="v1.Cmq..."
  python retrieve_frames_qwen3.py [options]

Options
-------
  --queries-dir    DIR   Query JSON files (default: ./frame_retriever_queries)
  --frames-dir     DIR   Dense PNG frames  (default: ./data/dense_frames)
  --output-dir     DIR   Results output    (default: ./frame_retrieval_results_qwen3)
  --cache-dir      DIR   Caption+embedding cache (default: ./qwen3_embed_cache)
  --embed-model    NAME  Nebius embedding model  (default: Qwen/Qwen3-Embedding)
  --caption-model  NAME  Nebius VLM for captions (default: Qwen/Qwen3-VL-72B-Instruct)
  --api-base       URL   Nebius API base URL
  --api-key        KEY   Nebius API key (or set NEBIUS_API_KEY)
  --top-k          INT   Top-K frames to return per query (default: 5)
  --embed-batch    INT   Texts per embedding API call (default: 32)
  --max-frames     INT   Cap frames per video (uniform sample). 0 = no cap (default: 0)
  --caption-only        Only run captioning step, skip embedding/retrieval
  --dry-run             Print plan without calling any API
"""

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai package not found.  pip install openai")

try:
    from PIL import Image
except ImportError:
    Image = None

# ── Default paths relative to this script ────────────────────────────────────
_EVAL_DIR = Path(__file__).parent
_NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts_from_frame_name(name: str) -> float:
    """frame_12.34.png  →  12.34"""
    try:
        return float(Path(name).stem.replace("frame_", ""))
    except ValueError:
        return 0.0


def _list_frame_paths(dense_frames_root: Path, video_id: str) -> list:
    """Return sorted list of frame PNGs for a video (sorted by timestamp)."""
    video_dir = dense_frames_root / video_id
    if not video_dir.is_dir():
        return []
    files = [
        f for f in video_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        and f.stem.startswith("frame_")
    ]
    files.sort(key=lambda p: _ts_from_frame_name(p.name))
    return files


def _uniform_sample(items: list, max_n: int) -> list:
    """Evenly subsample `items` to at most `max_n` entries."""
    n = len(items)
    if n <= max_n:
        return items
    indices = [int(round(i * (n - 1) / (max_n - 1))) for i in range(max_n)]
    seen, out = set(), []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            out.append(items[idx])
    return out


def _encode_image_b64(image_path: str, max_side: int = 768) -> str:
    """Load a PNG/JPEG, downscale if needed, return base64 JPEG string."""
    if Image is None:
        raise RuntimeError("Pillow is not installed.  pip install pillow")
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _cosine_sim(q: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """q: [D], matrix: [N, D] → scores [N]  (cosine similarity)."""
    q_n = q / (np.linalg.norm(q) + 1e-10)
    m_n = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return m_n @ q_n


def _path_hash(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def _load_json(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# API wrappers — retry on transient failures
# ─────────────────────────────────────────────────────────────────────────────

def _retry(fn, retries: int = 4, backoff: float = 5.0):
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                wait = backoff * (2 ** attempt)
                print(f"    [retry {attempt+1}/{retries-1}] {exc!r}  — waiting {wait:.0f}s")
                time.sleep(wait)
    raise last_exc


def caption_frame(client: OpenAI, image_path: str, model: str) -> str:
    """
    Send a single frame to the VLM and get a concise visual description.
    Prompt is tailored for math/diagram videos.
    """
    b64 = _encode_image_b64(image_path)
    prompt = (
        "Describe this video frame in one or two concise sentences. "
        "Focus on the main visual content: any mathematical diagrams, "
        "geometric shapes, equations, on-screen text, labels, colours, "
        "people, objects, or scene. Be specific and precise."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    def _call():
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=150,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()

    return _retry(_call)


def embed_texts_batch(
    client: OpenAI,
    texts: list,
    model: str,
    batch_size: int = 32,
) -> list:
    """Embed a list of strings in batches; returns list of float32 ndarrays."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        def _call(b=batch):
            resp = client.embeddings.create(model=model, input=b)
            resp.data.sort(key=lambda x: x.index)
            return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

        results.extend(_retry(_call))
    return results


def embed_single(client: OpenAI, text: str, model: str) -> np.ndarray:
    def _call():
        resp = client.embeddings.create(model=model, input=text)
        return np.array(resp.data[0].embedding, dtype=np.float32)
    return _retry(_call)


# ─────────────────────────────────────────────────────────────────────────────
# Disk cache
# ─────────────────────────────────────────────────────────────────────────────

class Cache:
    """
    Layout under cache_dir/:
      captions/  <md5(frame_abs_path)>.txt   — one caption per frame
      frames/    <video_id>.npz              — paths + embedding matrix
      queries/   <sha256(query_text)[:20].npy — per-query embedding
    """

    def __init__(self, cache_dir: Path):
        self.captions_dir = Path(cache_dir) / "captions"
        self.frames_dir   = Path(cache_dir) / "frames"
        self.queries_dir  = Path(cache_dir) / "queries"
        for d in (self.captions_dir, self.frames_dir, self.queries_dir):
            d.mkdir(parents=True, exist_ok=True)

    # Captions ----------------------------------------------------------------

    def load_caption(self, frame_abs_path: str) -> str | None:
        p = self.captions_dir / f"{_path_hash(frame_abs_path)}.txt"
        return p.read_text(encoding="utf-8").strip() if p.exists() else None

    def save_caption(self, frame_abs_path: str, caption: str):
        p = self.captions_dir / f"{_path_hash(frame_abs_path)}.txt"
        p.write_text(caption, encoding="utf-8")

    def caption_count(self) -> int:
        return len(list(self.captions_dir.glob("*.txt")))

    # Frame embeddings --------------------------------------------------------

    def load_frame_embeddings(self, video_id: str):
        """Returns (paths: list[str], embs: np.ndarray[N,D]) or (None, None)."""
        p = self.frames_dir / f"{video_id}.npz"
        if not p.exists():
            return None, None
        data = np.load(str(p), allow_pickle=True)
        return list(data["paths"]), data["embeddings"]

    def save_frame_embeddings(
        self, video_id: str, paths: list, embeddings: np.ndarray
    ):
        p = self.frames_dir / f"{video_id}.npz"
        np.savez_compressed(str(p), paths=np.array(paths), embeddings=embeddings)

    # Query embeddings --------------------------------------------------------

    def load_query_embedding(self, query: str) -> np.ndarray | None:
        p = self.queries_dir / f"{_text_hash(query)}.npy"
        return np.load(str(p)) if p.exists() else None

    def save_query_embedding(self, query: str, emb: np.ndarray):
        p = self.queries_dir / f"{_text_hash(query)}.npy"
        np.save(str(p), emb)


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def ensure_captions(
    frame_paths: list,   # list of Path objects
    client: OpenAI,
    caption_model: str,
    cache: Cache,
    dry_run: bool = False,
) -> list:
    """
    For every frame without a cached caption, calls the VLM and stores it.
    Returns list of caption strings aligned to frame_paths.
    """
    captions = []
    missing = [fp for fp in frame_paths if cache.load_caption(str(fp)) is None]

    if missing and not dry_run:
        print(f"    Captioning {len(missing)} new frames (model={caption_model})...")
    elif missing and dry_run:
        print(f"    [dry-run] Would caption {len(missing)} frames.")

    for fp in frame_paths:
        cap = cache.load_caption(str(fp))
        if cap is None:
            if not dry_run:
                cap = caption_frame(client, str(fp), caption_model)
                cache.save_caption(str(fp), cap)
                print(f"      ✓ {fp.name}: {cap[:80]}...")
            else:
                cap = f"[dry-run caption for {fp.name}]"
                cache.save_caption(str(fp), cap)
        captions.append(cap)
    return captions


def ensure_frame_embeddings(
    video_id: str,
    frame_paths: list,   # list of Path
    captions: list,      # aligned to frame_paths
    client: OpenAI,
    embed_model: str,
    embed_batch: int,
    cache: Cache,
    dry_run: bool = False,
) -> tuple:
    """
    Returns (path_strs: list[str], emb_matrix: np.ndarray[N, D]).
    Recomputes only when the cached path list doesn't match the current frames.
    """
    path_strs = [str(fp) for fp in frame_paths]

    cached_paths, cached_embs = cache.load_frame_embeddings(video_id)
    if cached_paths is not None and cached_paths == path_strs:
        print(f"    Frame embeddings loaded from cache ({len(cached_paths)} frames).")
        return cached_paths, cached_embs

    if dry_run:
        print(f"    [dry-run] Would embed {len(captions)} frame captions.")
        dummy = np.zeros((len(frame_paths), 1024), dtype=np.float32)
        return path_strs, dummy

    print(f"    Embedding {len(captions)} frame captions (model={embed_model})...")
    emb_list = embed_texts_batch(client, captions, embed_model, batch_size=embed_batch)
    emb_matrix = np.stack(emb_list, axis=0)   # [N, D]

    cache.save_frame_embeddings(video_id, path_strs, emb_matrix)
    print(f"    Saved frame embeddings ({emb_matrix.shape}).")
    return path_strs, emb_matrix


def retrieve(
    query: str,
    frame_paths: list,
    frame_embs: np.ndarray,
    client: OpenAI,
    embed_model: str,
    cache: Cache,
    top_k: int,
    dry_run: bool = False,
) -> list:
    """Returns list of {frame_path, timestamp, score} dicts, sorted by score."""
    q_emb = cache.load_query_embedding(query)
    if q_emb is None:
        if dry_run:
            q_emb = np.zeros(frame_embs.shape[1], dtype=np.float32)
        else:
            q_emb = embed_single(client, query, embed_model)
            cache.save_query_embedding(query, q_emb)

    scores = _cosine_sim(q_emb, frame_embs)          # [N]
    k = min(top_k, len(scores))
    top_idx = np.argsort(scores)[::-1][:k]

    return [
        {
            "frame_path": frame_paths[i],
            "timestamp": _ts_from_frame_name(Path(frame_paths[i]).name),
            "score": float(scores[i]),
        }
        for i in top_idx
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve dense frames for frame_retriever queries via Qwen3-Embedding (Nebius).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--queries-dir",   default=str(_EVAL_DIR / "frame_retriever_queries"))
    parser.add_argument("--frames-dir",    default=str(_EVAL_DIR / "data" / "dense_frames"))
    parser.add_argument("--output-dir",    default=str(_EVAL_DIR / "frame_retrieval_results_qwen3"))
    parser.add_argument("--cache-dir",     default=str(_EVAL_DIR / "qwen3_embed_cache"))
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-8B"),
        help="Nebius embedding model name",
    )
    parser.add_argument(
        "--caption-model",
        default=os.environ.get("CAPTION_MODEL", "Qwen/Qwen3-VL-72B-Instruct"),
        help="Nebius VLM for frame captioning",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("NEBIUS_API_BASE", _NEBIUS_BASE_URL),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NEBIUS_API_KEY", ""),
    )
    parser.add_argument("--top-k",       type=int, default=5)
    parser.add_argument("--embed-batch", type=int, default=32,
                        help="Texts per embeddings API call")
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Uniform-sample videos to at most N frames (0 = no cap)",
    )
    parser.add_argument(
        "--caption-only", action="store_true",
        help="Only run the captioning step; skip embedding and retrieval",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without calling any API",
    )
    args = parser.parse_args()

    if not args.api_key and not args.dry_run:
        sys.exit("Error: set --api-key or NEBIUS_API_KEY environment variable.")

    queries_dir = Path(args.queries_dir)
    frames_dir  = Path(args.frames_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.api_base, api_key=args.api_key or "EMPTY")
    cache  = Cache(Path(args.cache_dir))

    query_files = sorted(queries_dir.glob("*.json"))
    if not query_files:
        sys.exit(f"No query files found in {queries_dir}")

    print(f"Embed model  : {args.embed_model}")
    print(f"Caption model: {args.caption_model}")
    print(f"API base     : {args.api_base}")
    print(f"Videos       : {len(query_files)}")
    print(f"Top-K        : {args.top_k}")
    print(f"Cache dir    : {args.cache_dir}")
    if args.dry_run:
        print("*** DRY RUN — no real API calls will be made ***")

    for qf in query_files:
        video_id = qf.stem
        print(f"\n{'='*64}")
        print(f"Video: {video_id}")

        # ── load queries ────────────────────────────────────────────────────
        data = _load_json(qf)
        video_path = data.get("video_path", "")

        result_dirs_queries = {
            k: v for k, v in data.items()
            if k != "video_path" and isinstance(v, dict)
        }
        if not result_dirs_queries:
            print("  No queries — skipping.")
            continue

        # ── collect all unique queries across result dirs ───────────────────
        all_unique_queries: set = set()
        for qmap in result_dirs_queries.values():
            all_unique_queries.update(qmap.values())
        print(f"  Unique queries : {len(all_unique_queries)}")

        # ── load dense frames ───────────────────────────────────────────────
        all_frames = _list_frame_paths(frames_dir, video_id)
        if not all_frames:
            print(f"  No dense frames found under {frames_dir / video_id} — skipping.")
            continue

        if args.max_frames and len(all_frames) > args.max_frames:
            all_frames = _uniform_sample(all_frames, args.max_frames)
            print(f"  Frames (capped): {len(all_frames)}")
        else:
            print(f"  Frames         : {len(all_frames)}")

        # ── step 1: ensure captions ─────────────────────────────────────────
        captions = ensure_captions(
            all_frames, client, args.caption_model, cache, dry_run=args.dry_run
        )

        if args.caption_only:
            print("  --caption-only: skipping embedding and retrieval.")
            continue

        # ── step 2: ensure frame embeddings ────────────────────────────────
        try:
            fp_strs, frame_embs = ensure_frame_embeddings(
                video_id, all_frames, captions,
                client, args.embed_model, args.embed_batch,
                cache, dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"  ERROR building frame embeddings: {exc}")
            continue

        # ── step 3: retrieve for every query ───────────────────────────────
        output_record = {
            "video_path": video_path,
            "embed_model": args.embed_model,
            "caption_model": args.caption_model,
            "num_frames": len(fp_strs),
            "results": {},
        }

        for result_dir_name, qmap in result_dirs_queries.items():
            output_record["results"][result_dir_name] = {}
            for q_key, q_text in qmap.items():
                if not str(q_text).strip():
                    continue
                try:
                    top_frames = retrieve(
                        q_text, fp_strs, frame_embs,
                        client, args.embed_model, cache,
                        args.top_k, dry_run=args.dry_run,
                    )
                    output_record["results"][result_dir_name][q_key] = {
                        "query": q_text,
                        "top_frames": top_frames,
                    }
                    best = top_frames[0]
                    print(
                        f"  [{result_dir_name}] {q_key}: "
                        f"top frame @ t={best['timestamp']:.2f}s  "
                        f"score={best['score']:.4f}"
                    )
                except Exception as exc:
                    print(f"  ERROR [{result_dir_name}] {q_key}: {exc}")
                    output_record["results"][result_dir_name][q_key] = {
                        "query": q_text,
                        "error": str(exc),
                    }

        out_path = output_dir / f"{video_id}.json"
        _save_json(output_record, out_path)
        print(f"  ✓ Saved: {out_path}")

    print(f"\n✓ Done.  Results → {output_dir}/")
    print(f"         Cache   → {args.cache_dir}/")


if __name__ == "__main__":
    main()
