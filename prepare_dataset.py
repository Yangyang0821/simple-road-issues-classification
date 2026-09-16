"""Prepare the road-issues dataset into a short-path, training-ready layout.

Problem
    kagglehub extracts into a deeply nested cache path:
        dataset/datasets/<owner>/<slug>/versions/<n>/data/<group>/<class>/<file>.jpg
    Combined with this dataset's verbose class names (e.g. "Public Cleanliness +
    Environmental Issues/Littering Garbage on Public Places Issues") and Roboflow's
    long filenames, paths exceed Windows' 260-character MAX_PATH limit and zipfile
    raises FileNotFoundError. Lifting that limit needs admin rights, so we don't.

Solution
    1. Prefix every write/move with \\?\ via long_path() to bypass MAX_PATH.
    2. Own the final layout: short class slugs plus truncated filenames, so paths
       stay far below 260 and later reads (PIL, DataLoader) need no prefix.
    3. Keep outputs and cache under dataset/ and use manifest.json as a completion
       marker, so re-runs never re-download or re-extract.

Layout
    dataset/cache/road-issues-v2.zip   archive cache (kept, so rebuilds need no download)
    dataset/raw/<class_slug>/*.jpg     what training actually reads
    dataset/raw/manifest.json          completion marker + class map + rename log

Usage
    python prepare_dataset.py            # no-op once prepared
    python prepare_dataset.py --force    # ignore the marker and rebuild raw/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from config import Dataset_info, Hyperparameters, PROJECT_DIR

DATASET_DIR = PROJECT_DIR / "dataset"
CACHE_DIR = DATASET_DIR / "cache"
RAW_DIR = DATASET_DIR / "raw"
ARCHIVE_PATH = CACHE_DIR / "road-issues-v2.zip"
MANIFEST_PATH = RAW_DIR / "manifest.json"
TMP_CACHE = CACHE_DIR / "_kagglehub"

# Original Kaggle class name -> short slug used on disk.
CLASS_SLUGS: dict[str, str] = {
    "Broken Road Sign Issues": "broken_road_sign",
    "Damaged Road issues": "damaged_road",
    "Illegal Parking Issues": "illegal_parking",
    "Pothole Issues": "pothole",
    "Littering Garbage on Public Places Issues": "littering",
    "Vandalism Issues": "vandalism",
    # Multi-label category: files are kept but excluded from the 6-class task.
    "Mixed Issues": "mixed"
}

# Classes used for training; position is the label index (0..5).
TRAIN_CLASSES: tuple[str, ...] = (
    "broken_road_sign",
    "damaged_road",
    "illegal_parking",
    "pothole",
    "littering",
    "vandalism"
)

# Filename cap. The dataset/raw/<slug>/ prefix is ~60 chars, so the full path
# stays around 140 -- well under the 260 limit even if the project moves deeper.
MAX_FILENAME = 80


def long_path(path: str | os.PathLike[str]) -> str:
    r"""Return `path` prefixed with \\?\ on Windows to bypass the MAX_PATH limit.

    The prefix makes Win32 skip path-length validation. It needs no admin rights
    and no registry change, but requires an absolute, backslash-only path --
    abspath() guarantees both. Returned unchanged on other platforms.
    """
    resolved = os.path.abspath(str(path))
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):  # UNC network path
        return "\\\\?\\UNC\\" + resolved[2:]
    return "\\\\?\\" + resolved


def shorten_filename(name: str) -> str:
    """Truncate an over-long filename, appending a hash to keep it unique."""
    if len(name) <= MAX_FILENAME:
        return name
    stem, ext = os.path.splitext(name)
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    keep = MAX_FILENAME - len(ext) - len(digest) - 1
    return f"{stem[:keep]}_{digest}{ext}"


def remove_tree(path: Path) -> None:
    """Recursively delete `path`, long-path safe."""
    if not path.exists():
        return

    # Clear the read-only attribute and retry; re-raise anything else so the
    # caller decides how to report it.
    def _on_error(func, failed_path, exc_info):  # noqa: ANN001
        if isinstance(exc_info[1], PermissionError):
            os.chmod(failed_path, 0o600)
            func(failed_path)
            return
        raise exc_info[1]

    shutil.rmtree(long_path(path), onerror=_on_error)


def read_manifest() -> dict | None:
    if not MANIFEST_PATH.exists():
        return None
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def manifest_is_valid(manifest: dict | None) -> bool:
    """Check the per-class counts in the manifest against what is on disk."""
    if not manifest:
        return False
    for slug, info in manifest.get("classes", {}).items():
        class_dir = RAW_DIR / slug
        if not class_dir.is_dir():
            return False
        if len(list(class_dir.glob("*.jpg"))) != info["count"]:
            return False
    return True


def write_manifest(counts: dict[str, int], renamed: dict[str, str], source: str) -> dict:
    slug_to_original = {slug: original for original, slug in CLASS_SLUGS.items()}
    manifest = {
        "dataset": Dataset_info.dataset_name,
        "version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "layout": "raw/<class_slug>/<filename>.jpg",
        "train_classes": list(TRAIN_CLASSES),
        "classes": {
            slug: {
                "original_name": slug_to_original[slug],
                "index": TRAIN_CLASSES.index(slug) if slug in TRAIN_CLASSES else None,
                "count": count,
            }
            for slug, count in sorted(counts.items())
        },
        "total_images": sum(counts.values()),
        # Only files above MAX_FILENAME are renamed; keep the mapping for traceability.
        "renamed_files": renamed,
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def download() -> tuple[Path | None, Path | None]:
    """Fetch the dataset via kagglehub. Returns (archive, extracted_dir).

    kagglehub downloads the zip first, then extracts it itself:
      - On Windows that extraction fails on MAX_PATH, but the zip is already
        complete, so catching the error still leaves us a usable archive.
      - Elsewhere extraction succeeds and kagglehub deletes the zip, so we get
        an extracted directory instead.
    Exactly one of the two is returned; (None, None) means the fetch failed.
    """
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed; run: pip install kagglehub")
        return None, None
    TMP_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(TMP_CACHE)
    print(f"Downloading {Dataset_info.dataset_name} from Kaggle ...")
    extracted = None
    try:
        extracted = Path(kagglehub.dataset_download(Dataset_info.dataset_name))
    except (FileNotFoundError, OSError) as exc:
        # Expected on Windows: kagglehub's auto-extraction hit MAX_PATH.
        print(f"  kagglehub extraction failed ({type(exc).__name__}); extracting it ourselves")
    for candidate in sorted(TMP_CACHE.rglob("*.archive")):
        if zipfile.is_zipfile(candidate):
            os.replace(long_path(candidate), long_path(ARCHIVE_PATH))
            return ARCHIVE_PATH, None
    return None, extracted


def _target_path(filename: str, class_name: str, renamed: dict[str, str]) -> Path | None:
    """Map one source file to its place in raw/, or None if its class is unknown."""
    slug = CLASS_SLUGS.get(class_name)
    if slug is None:
        return None
    safe_name = shorten_filename(filename)
    if safe_name != filename:
        renamed[f"{slug}/{safe_name}"] = filename
    class_dir = RAW_DIR / slug
    class_dir.mkdir(parents=True, exist_ok=True)
    return class_dir / safe_name


def build_from_zip(archive: Path) -> dict[str, str]:
    """Extract the archive into raw/, re-mapping classes and filenames."""
    renamed: dict[str, str] = {}
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        total = len(members)
        print(f"Extracting {total} files -> {RAW_DIR}")
        for done, member in enumerate(members, start=1):
            name = Path(member.filename)
            target = _target_path(name.name, name.parent.name, renamed)
            if target is None:
                continue
            with zf.open(member) as src, open(long_path(target), "wb") as dst:
                shutil.copyfileobj(src, dst)
            if done % 1000 == 0 or done == total:
                print(f"  {done}/{total}")
    return renamed


def build_from_dir(source: Path) -> dict[str, str]:
    """Move an already-extracted tree into raw/ (used when kagglehub extracted it)."""
    renamed: dict[str, str] = {}
    print(f"Moving extracted files -> {RAW_DIR}")
    for src in source.rglob("*.jpg"):
        target = _target_path(src.name, src.parent.name, renamed)
        if target is None:
            continue
        os.replace(long_path(src), long_path(target))
    return renamed


def count_images() -> dict[str, int]:
    """Images per class as they actually exist on disk."""
    return {d.name: len(list(d.glob("*.jpg"))) for d in sorted(RAW_DIR.iterdir()) if d.is_dir()}


def prepare(force: bool = False) -> Path:
    """Ensure dataset/raw/ is ready and return its path. A no-op once prepared."""
    manifest = read_manifest()
    if not force and manifest_is_valid(manifest):
        print(f"Dataset ready: {RAW_DIR} ({manifest['total_images']} images, nothing to do)")
        return RAW_DIR
    if force and RAW_DIR.exists():
        print("--force: removing raw/ and rebuilding")
        remove_tree(RAW_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    archive = ARCHIVE_PATH if ARCHIVE_PATH.exists() else None
    if archive is not None:
        source = f"cached archive ({archive.name})"
        renamed = build_from_zip(archive)
    else:
        archive, extracted = download()
        if archive is not None:
            source = f"kaggle download -> archive ({archive.name})"
            renamed = build_from_zip(archive)
        elif extracted is not None:
            source = "kaggle download -> extracted tree"
            renamed = build_from_dir(extracted)
        else:
            msg = "Could not obtain the dataset. Check your Kaggle credentials."
            raise RuntimeError(msg)
    # The temp cache is disposable either way; a failed cleanup is not fatal.
    try:
        remove_tree(TMP_CACHE)
    except OSError as exc:
        print(f"Note: could not remove {TMP_CACHE.name}/ ({exc.strerror}); remove it manually")

    manifest = write_manifest(count_images(), renamed, source)
    print(f"\nDone: {manifest['total_images']} images across {len(manifest['classes'])} classes")
    if renamed:
        print(f"{len(renamed)} over-long filenames shortened (mapping in manifest.json)")
    print(f"Archive cache: {ARCHIVE_PATH if ARCHIVE_PATH.exists() else '(none)'}")
    return RAW_DIR


def get_class_to_idx() -> dict[str, int]:
    """Class slug -> label index, for the six training classes only."""
    return {slug: idx for idx, slug in enumerate(TRAIN_CLASSES)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the road-issues dataset (short-path layout)")
    parser.add_argument("--force", action="store_true", help="ignore the marker and rebuild raw/")
    args = parser.parse_args()
    prepare(force=args.force)
