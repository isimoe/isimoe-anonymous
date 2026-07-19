import argparse
import json
import os
from typing import Any, Dict, List

import h5py
import numpy as np


GENRE_NAMES = [
    "Action",
    "Adventure",
    "Animation",
    "Biography",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "Film-Noir",
    "History",
    "Horror",
    "Music",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Short",
    "Sport",
    "Thriller",
    "War",
    "Western",
]


def _decode_imdb_id(x: Any) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.generic):
        return str(x.item())
    return str(x)


def _labels_to_active_genres(labels_row: np.ndarray, threshold: float = 0.5) -> List[str]:
    row = np.asarray(labels_row).reshape(-1)
    idxs = np.where(row > threshold)[0].tolist()
    out: List[str] = []
    for i in idxs:
        if 0 <= i < len(GENRE_NAMES):
            out.append(GENRE_NAMES[i])
        else:
            out.append(str(i))
    return out


def _to_uint8(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a)
    if x.dtype.kind in ("u", "i"):
        x = x.astype(np.float32)
        if x.min() < 0:
            x = x - float(x.min())
        if x.max() > 255:
            x = x / (float(x.max()) + 1e-8) * 255.0
        return np.clip(x, 0, 255).astype(np.uint8)
    if x.dtype.kind == "f":
        amin = float(x.min())
        amax = float(x.max())
        if 0.0 <= amin and amax <= 1.0:
            return np.clip(x * 255.0, 0, 255).astype(np.uint8)
        if -1.1 <= amin and amax <= 1.1:
            y = (x + 1.0) / 2.0
            return np.clip(y * 255.0, 0, 255).astype(np.uint8)
        y = (x - amin) / (amax - amin + 1e-8)
        return np.clip(y * 255.0, 0, 255).astype(np.uint8)
    return x.astype(np.uint8)


def _save_png_chw(img_chw: np.ndarray, out_png: str) -> None:
    from PIL import Image

    x = _to_uint8(img_chw)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D CHW array, got shape {x.shape}")
    if x.shape[0] not in (1, 3, 4):
        raise ValueError(f"Expected C in (1,3,4), got {x.shape[0]}")
    x_hwc = np.transpose(x, (1, 2, 0))
    if x_hwc.shape[2] == 1:
        im = Image.fromarray(x_hwc[:, :, 0], mode="L")
    elif x_hwc.shape[2] == 3:
        im = Image.fromarray(x_hwc, mode="RGB")
    else:
        im = Image.fromarray(x_hwc, mode="RGBA")
    im.save(out_png)


def dump_mmimdb(
    hdf5_path: str,
    output_dir: str,
    max_samples: int = 0,
    threshold: float = 0.5,
    dump_images: bool = True,
    dump_language: bool = True,
    dump_labels: bool = True,
    dump_sequences_token_ids: bool = True,
) -> Dict[str, Any]:
    """Export images, language features, token sequences, and labels from HDF5."""
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    language_dir = os.path.join(output_dir, "language")
    labels_dir = os.path.join(output_dir, "labels")
    if dump_images:
        os.makedirs(images_dir, exist_ok=True)
    if dump_language:
        os.makedirs(language_dir, exist_ok=True)
    if dump_labels:
        os.makedirs(labels_dir, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        imdb_ids = f["imdb_ids"][:]
        n_total = int(imdb_ids.shape[0])
        n = n_total if max_samples <= 0 else min(n_total, max_samples)

        images = f["images"] if dump_images else None
        language = f["features"] if dump_language else None
        genres = f["genres"] if dump_labels else None
        sequences = f["sequences"] if dump_sequences_token_ids else None

        for i in range(n):
            imdb_id = _decode_imdb_id(imdb_ids[i])

            if dump_images and images is not None:
                out_png = os.path.join(images_dir, f"{imdb_id}.png")
                img = np.asarray(images[i])  # expected CHW
                _save_png_chw(img, out_png)

            if dump_language and language is not None:
                lang = np.asarray(language[i])
                np.save(os.path.join(language_dir, f"{imdb_id}.npy"), lang)
                # short preview
                flat = lang.reshape(-1)
                with open(os.path.join(language_dir, f"{imdb_id}.txt"), "w", encoding="utf-8") as tf:
                    tf.write(f"imdb_id: {imdb_id}\n")
                    tf.write(f"shape: {tuple(lang.shape)}\n")
                    tf.write("head(50):\n")
                    for j, v in enumerate(flat[:50].tolist()):
                        tf.write(f"{j:02d}: {float(v):.8f}\n")

                # Also dump the token-id sequence as readable text, if available.
                # Note: this is NOT real English words unless an idx2word vocabulary is provided.
                if sequences is not None:
                    try:
                        seq = np.asarray(sequences[i]).reshape(-1)
                        with open(
                            os.path.join(language_dir, f"{imdb_id}_token_ids.txt"),
                            "w",
                            encoding="utf-8",
                        ) as sf:
                            sf.write(f"imdb_id: {imdb_id}\n")
                            sf.write(f"token_ids_len: {int(seq.shape[0])}\n")
                            sf.write("token_ids:\n")
                            sf.write(" ".join(str(int(x)) for x in seq.tolist()) + "\n")
                    except Exception:
                        pass

            if dump_labels and genres is not None:
                y = np.asarray(genres[i]).reshape(-1)
                active = _labels_to_active_genres(y, threshold=threshold)
                with open(os.path.join(labels_dir, f"{imdb_id}.txt"), "w", encoding="utf-8") as tf:
                    tf.write(f"imdb_id: {imdb_id}\n")
                    tf.write(f"active_genres(threshold>{threshold}):\n")
                    if active:
                        for g in active:
                            tf.write(f"- {g}\n")
                    else:
                        tf.write("(none)\n")
                with open(os.path.join(labels_dir, f"{imdb_id}.json"), "w", encoding="utf-8") as jf:
                    json.dump(
                        {
                            "imdb_id": imdb_id,
                            "labels_numeric": y.astype(int).tolist(),
                            "active_genres": active,
                            "threshold": threshold,
                        },
                        jf,
                        ensure_ascii=False,
                        indent=2,
                    )

    return {
        "n_total": n_total,
        "exported_n": n,
        "exported": {
            "images": dump_images,
            "language": dump_language,
            "labels": dump_labels,
            "sequences_token_ids": dump_sequences_token_ids,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 multimodal_imdb.hdf5 解析并导出：图片(images)、语言(features+sequences)、标签(genres)。"
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        default="data/mm-imdb/multimodal_imdb.hdf5",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/mm-imdb/output",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no_images", action="store_true")
    parser.add_argument("--no_language", action="store_true")
    parser.add_argument("--no_labels", action="store_true")
    parser.add_argument(
        "--no_sequences",
        action="store_true",
        help="Do NOT dump per-sample token-id sequences into language/*.txt",
    )
    args = parser.parse_args()

    summary = dump_mmimdb(
        hdf5_path=args.hdf5_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        threshold=args.threshold,
        dump_images=not bool(args.no_images),
        dump_language=not bool(args.no_language),
        dump_labels=not bool(args.no_labels),
        dump_sequences_token_ids=not bool(args.no_sequences),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
