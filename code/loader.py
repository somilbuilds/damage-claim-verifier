"""
loader.py — Data loading and image preprocessing for the damage claim pipeline.

Reads CSVs into structured dicts and encodes images to base64 JPEG,
resizing if the longest side exceeds 1568px to stay within token limits.
"""

import base64
import csv
import io
import logging
import os
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

# Maximum pixel dimension on the longest side before resizing.
MAX_IMAGE_SIDE = 1568

# Resolve the dataset folder relative to this file's location.
# code/loader.py -> code/ -> repo root -> dataset/
_CODE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CODE_DIR.parent
DATASET_DIR = _REPO_ROOT / "dataset"


def load_claims(csv_path: str | Path) -> list[dict]:
    """Read a claims CSV (sample or test) into a list of row dicts.

    Works for both sample_claims.csv (has output columns) and
    claims.csv (input columns only).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Claims CSV not found: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    logger.info("Loaded %d claims from %s", len(rows), csv_path.name)
    return rows


def load_user_history(csv_path: str | Path | None = None) -> dict[str, dict]:
    """Read user_history.csv into a dict keyed by user_id.

    Returns:
        { "user_001": { "past_claim_count": "2", ... }, ... }
    """
    if csv_path is None:
        csv_path = DATASET_DIR / "user_history.csv"
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"User history CSV not found: {csv_path}")

    history = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["user_id"]
            history[uid] = dict(row)

    logger.info("Loaded history for %d users from %s", len(history), csv_path.name)
    return history


def load_evidence_requirements(csv_path: str | Path | None = None) -> dict[str, list[dict]]:
    """Read evidence_requirements.csv into a dict keyed by claim_object.

    Requirements with claim_object="all" are stored under the "all" key.
    When looking up requirements for a specific object, callers should
    combine the object-specific list with the "all" list.

    Returns:
        {
            "all": [ { "requirement_id": ..., "applies_to": ..., ... }, ... ],
            "car": [ ... ],
            "laptop": [ ... ],
            "package": [ ... ],
        }
    """
    if csv_path is None:
        csv_path = DATASET_DIR / "evidence_requirements.csv"
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Evidence requirements CSV not found: {csv_path}")

    reqs: dict[str, list[dict]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = row["claim_object"]
            reqs.setdefault(obj, []).append(dict(row))

    logger.info(
        "Loaded evidence requirements: %s",
        {k: len(v) for k, v in reqs.items()},
    )
    return reqs


def get_requirements_for_object(
    reqs: dict[str, list[dict]], claim_object: str
) -> list[dict]:
    """Return combined requirements: object-specific + 'all'."""
    result = list(reqs.get("all", []))
    if claim_object != "all":
        result.extend(reqs.get(claim_object, []))
    return result


def _resize_image(img: Image.Image) -> Image.Image:
    """Resize if the longest side exceeds MAX_IMAGE_SIDE, preserving aspect ratio."""
    w, h = img.size
    longest = max(w, h)
    if longest <= MAX_IMAGE_SIDE:
        return img

    scale = MAX_IMAGE_SIDE / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def _image_to_base64(img: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded JPEG string."""
    # Convert RGBA/palette to RGB for JPEG compatibility.
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_images(
    image_paths_str: str,
    dataset_dir: str | Path | None = None,
) -> list[dict]:
    """Load, resize, and base64-encode images from a semicolon-separated path string.

    Args:
        image_paths_str: e.g. "images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg"
        dataset_dir: Root directory that image paths are relative to.

    Returns:
        List of dicts, one per successfully loaded image:
        [
            {
                "image_id": "img_1",
                "path": "images/test/case_001/img_1.jpg",
                "base64": "<base64 string>",
                "mime_type": "image/jpeg",
                "width": 800,
                "height": 600,
            },
            ...
        ]
    """
    if dataset_dir is None:
        dataset_dir = DATASET_DIR
    dataset_dir = Path(dataset_dir)

    raw_paths = [p.strip() for p in image_paths_str.split(";") if p.strip()]
    results = []

    for rel_path in raw_paths:
        abs_path = dataset_dir / rel_path

        # Extract image_id: filename without extension.
        image_id = Path(rel_path).stem

        if not abs_path.exists():
            logger.warning("Image file not found, skipping: %s", abs_path)
            continue

        try:
            img = Image.open(abs_path)
            img = _resize_image(img)
            b64 = _image_to_base64(img)

            results.append(
                {
                    "image_id": image_id,
                    "path": rel_path,
                    "base64": b64,
                    "mime_type": "image/jpeg",
                    "width": img.size[0],
                    "height": img.size[1],
                }
            )
        except Exception as e:
            logger.warning("Failed to load image %s: %s", abs_path, e)
            continue

    if not results:
        logger.warning("No images loaded from paths: %s", image_paths_str)

    return results
