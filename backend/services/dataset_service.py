"""Dataset service responsible for discovering and describing local datasets."""
from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

from config.config import settings
from backend.models.schemas import DatasetInfo

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_EXTENSIONS = {".txt"}
DEFAULT_SPLITS = ("train", "val", "test")


class DatasetService:
    """Provide cached access to datasets stored on disk."""

    def __init__(self) -> None:
        self._cache: List[DatasetInfo] = []
        self._cache_timestamp: float = 0.0
        self._lock = threading.Lock()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def list_datasets(self, force_refresh: bool = False) -> List[DatasetInfo]:
        """Return cached dataset list, refreshing when required."""
        if not force_refresh:
            with self._lock:
                if (
                    self._cache
                    and (time.time() - self._cache_timestamp) < settings.DATASET_CACHE_TTL
                ):
                    return [item.copy(deep=True) for item in self._cache]

        datasets: List[DatasetInfo] = []
        visited: Set[Path] = set()

        for root in settings.dataset_search_paths:
            if not root.exists():
                continue

            for dataset_dir in self._discover_dataset_dirs(root):
                resolved = dataset_dir.resolve()
                if resolved in visited:
                    continue
                visited.add(resolved)

                info = self._build_dataset_info(resolved)
                if info:
                    datasets.append(info)

        datasets.sort(key=lambda item: item.name.lower())

        with self._lock:
            self._cache = datasets
            self._cache_timestamp = time.time()

        return [item.copy(deep=True) for item in datasets]

    def refresh(self) -> List[DatasetInfo]:
        """Force a rescan of all dataset locations."""
        return self.list_datasets(force_refresh=True)

    def clear_cache(self) -> None:
        """Reset cached dataset information."""
        with self._lock:
            self._cache = []
            self._cache_timestamp = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _discover_dataset_dirs(self, root: Path) -> List[Path]:
        """Identify dataset directories starting from a root path."""
        results: List[Path] = []

        if self._is_dataset_dir(root):
            results.append(root)

        for child in root.iterdir():
            if not child.is_dir():
                continue
            if self._is_dataset_dir(child):
                results.append(child)

        # Additionally search for nested data.yaml files (depth-limited)
        for yaml_path in root.glob("**/data.yaml"):
            candidate = yaml_path.parent
            if candidate not in results and self._is_dataset_dir(candidate):
                results.append(candidate)

        return results

    def _is_dataset_dir(self, directory: Path) -> bool:
        if not directory.is_dir():
            return False

        if (directory / "data.yaml").exists() or (directory / "dataset.yaml").exists():
            return True

        images_dir = directory / "images"
        labels_dir = directory / "labels"
        if images_dir.exists() and labels_dir.exists():
            if any((images_dir / split).exists() for split in DEFAULT_SPLITS):
                return True
            # Allow datasets without split folders but with images present
            if any(child.suffix.lower() in IMAGE_EXTENSIONS for child in images_dir.iterdir() if child.is_file()):
                return True

        return False

    def _build_dataset_info(self, dataset_dir: Path) -> Optional[DatasetInfo]:
        try:
            metadata_path = self._choose_metadata_file(dataset_dir)
            metadata: Dict[str, object] = {}
            if metadata_path:
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = yaml.safe_load(handle) or {}

            classes = self._extract_classes(dataset_dir, metadata)
            split_counts, total_images = self._count_images(dataset_dir)

            return DatasetInfo(
                name=dataset_dir.name,
                path=str(dataset_dir),
                num_images=total_images,
                num_classes=len(classes),
                classes=classes,
                split=split_counts,
                created_at=datetime.fromtimestamp(dataset_dir.stat().st_ctime),
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            print(f"[DatasetService] Failed to read dataset info for {dataset_dir}: {exc}")
            return None

    @staticmethod
    def _choose_metadata_file(dataset_dir: Path) -> Optional[Path]:
        candidates = [dataset_dir / "data.yaml", dataset_dir / "dataset.yaml"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _extract_classes(self, dataset_dir: Path, metadata: Dict[str, object]) -> List[str]:
        names = metadata.get("names") if isinstance(metadata, dict) else None

        if isinstance(names, dict):
            try:
                sorted_items = sorted(names.items(), key=lambda item: int(item[0]))
                classes = [str(value) for _, value in sorted_items]
            except Exception:
                classes = [str(value) for value in names.values()]
            if classes:
                return classes
        elif isinstance(names, list):
            classes = [str(value) for value in names if value is not None]
            if classes:
                return classes

        class_ids = self._collect_label_class_ids(dataset_dir)
        if class_ids:
            return [f"class_{idx}" for idx in sorted(class_ids)]

        return []

    def _collect_label_class_ids(self, dataset_dir: Path, limit: int = 400) -> Set[int]:
        labels_root = dataset_dir / "labels"
        if not labels_root.exists():
            return set()

        class_ids: Set[int] = set()
        processed_files = 0

        for path in labels_root.rglob("*.txt"):
            processed_files += 1
            if processed_files > limit and class_ids:
                break
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        first_token = stripped.split()[0]
                        class_ids.add(int(float(first_token)))
            except Exception:
                continue

        return class_ids

    def _count_images(self, dataset_dir: Path) -> (Dict[str, int], int):
        images_root = dataset_dir / "images"
        split_counts: Dict[str, int] = {}
        total_images = 0

        if images_root.exists():
            for split in DEFAULT_SPLITS:
                split_dir = images_root / split
                if split_dir.exists():
                    count = self._count_files(split_dir, IMAGE_EXTENSIONS)
                    if count:
                        split_counts[split] = count
                        total_images += count

            if not split_counts:
                total_images = self._count_files(images_root, IMAGE_EXTENSIONS)
        else:
            total_images = self._count_files(dataset_dir, IMAGE_EXTENSIONS)

        if not split_counts and total_images:
            split_counts["all"] = total_images

        return split_counts, total_images

    @staticmethod
    def _count_files(directory: Path, extensions: Set[str]) -> int:
        if not directory.exists():
            return 0
        count = 0
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                count += 1
        return count


dataset_service = DatasetService()
