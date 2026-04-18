from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Iterator

import cv2
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

try:
    from .feature_extractor import FeatureExtractor
    from . import unique_colors
except ImportError:
    from feature_extractor import FeatureExtractor
    import unique_colors

COORD_COLUMNS = ["coor_x1_20x", "coor_y1_20x", "coor_x2_20x", "coor_y2_20x"]
_TILE_COORD_RE = re.compile(r"^\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$")


def parse_tile_coords(tile_path: str) -> tuple[int, int, int, int]:
    stem = os.path.splitext(os.path.basename(tile_path))[0]
    match = _TILE_COORD_RE.match(stem)
    if not match:
        raise ValueError(f"Tile filename does not encode coordinates: {tile_path}")
    return tuple(int(match.group(i)) for i in range(1, 5))


def list_saved_tile_paths(tiles_dir: str) -> list[str]:
    if not os.path.isdir(tiles_dir):
        return []

    out: list[str] = []
    for name in os.listdir(tiles_dir):
        tile_path = os.path.join(tiles_dir, name)
        if not os.path.isfile(tile_path):
            continue
        if os.path.splitext(name)[1].lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        out.append(tile_path)

    out.sort(key=parse_tile_coords)
    return out


def _batched(items: list[str], batch_size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


@dataclass(slots=True)
class HAVOCConfig:
    k_vals: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9])
    tile_size: int = 512
    min_tissue_amt: float = 0.2
    batch_size: int = 8

    def __post_init__(self):
        if self.tile_size % 256 != 0:
            raise ValueError("tile_size must multiple of 256")
        if not (0.0 <= self.min_tissue_amt <= 1.0):
            raise ValueError("min_tissue_amt must be between 0 and 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


class HAVOC:
    def __init__(self, havoc_config: HAVOCConfig, feature_extractor: FeatureExtractor | None = None):
        self.config = havoc_config
        self.feature_extractor = feature_extractor or FeatureExtractor()

    def run(self, tiles_dir: str, progress_callback=None, slide_name: str | None = None) -> pd.DataFrame:
        return self.run_tiles(
            tiles_dir,
            progress_callback=progress_callback,
            slide_name=slide_name,
        )

    def run_tiles(self, tiles_dir: str, progress_callback=None, slide_name: str | None = None) -> pd.DataFrame:
        tile_paths = list_saved_tile_paths(tiles_dir)
        if not tile_paths:
            return pd.DataFrame(columns=COORD_COLUMNS + self._cluster_columns())

        coords: list[tuple[int, int, int, int]] = []
        dlfvs: list[np.ndarray] = []
        total_tiles = len(tile_paths)

        if progress_callback is not None:
            prefix = f"{slide_name}: " if slide_name else ""
            progress_callback(f"{prefix}HAVOC 0.0%")

        processed = 0
        for batch_paths in _batched(tile_paths, self.config.batch_size):
            tiles = []
            for tile_path in batch_paths:
                tile = cv2.imread(tile_path, cv2.IMREAD_COLOR)
                if tile is None:
                    raise RuntimeError(f"Failed to read tile image: {tile_path}")
                tiles.append(tile)
                coords.append(parse_tile_coords(tile_path))

            batch = np.stack(tiles, axis=0)
            dlfvs.append(self.feature_extractor.process(batch))
            processed += len(batch_paths)

            if progress_callback is not None:
                prefix = f"{slide_name}: " if slide_name else ""
                progress_callback(f"{prefix}HAVOC {(processed / total_tiles) * 100.0:.1f}%")
        feature_cols = [str(x) for x in range(1, self.feature_extractor.num_features + 1)]
        feature_df = pd.DataFrame(np.concatenate(dlfvs, axis=0), columns=feature_cols)
        cluster_info_df = self.create_cluster_info_df(feature_df)
        coord_df = pd.DataFrame(coords, columns=COORD_COLUMNS)

        return pd.concat([coord_df.reset_index(drop=True), cluster_info_df.reset_index(drop=True)], axis=1)

    def _cluster_columns(self) -> list[str]:
        cols: list[str] = []
        for k in sorted(set(int(k) for k in self.config.k_vals)):
            cols.extend(
                [
                    f"k{k}",
                    f"k{k}_color",
                    f"k{k}_color_r",
                    f"k{k}_color_g",
                    f"k{k}_color_b",
                ]
            )
        return cols

    def create_cluster_info_df(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        k_vals = sorted(set(int(k) for k in self.config.k_vals))
        if not k_vals:
            return pd.DataFrame(index=feature_df.index)
        if feature_df.empty:
            return pd.DataFrame(columns=self._cluster_columns())

        num_rows = len(feature_df)
        color_gen = unique_colors.next_color_generator(scaled=False, mode="rgb", shuffle=False)

        if num_rows == 1:
            only_color = next(color_gen)
            dfs = []
            for k in k_vals:
                temp_df = pd.DataFrame({f"k{k}": [0]})
                temp_df[f"k{k}_color"] = [only_color["name"]]
                temp_df[[f"k{k}_color_r", f"k{k}_color_g", f"k{k}_color_b"]] = np.array([only_color["val"]])
                dfs.append(temp_df)
            return pd.concat(dfs, axis=1)

        feature_cols = [str(x) for x in range(1, self.feature_extractor.num_features + 1)]
        self.Z = linkage(feature_df[feature_cols].to_numpy(dtype=np.float32), method="ward")

        labels_per_k: dict[int, np.ndarray] = {}
        for k in k_vals:
            effective_k = min(k, num_rows)
            labels_per_k[k] = fcluster(self.Z, t=effective_k, criterion="maxclust") - 1

        cluster_color: dict[tuple[int, int], dict] = {}

        k_prev = k_vals[0]
        labels_prev = labels_per_k[k_prev]

        counts_prev = np.bincount(labels_prev, minlength=max(1, labels_prev.max() + 1))
        cluster_ids_sorted = np.argsort(counts_prev)[::-1]

        for cid in cluster_ids_sorted:
            cluster_color[(k_prev, int(cid))] = next(color_gen)

        for k in k_vals[1:]:
            labels_curr = labels_per_k[k]
            counts_curr = np.bincount(labels_curr, minlength=max(1, labels_curr.max() + 1))

            parent_for_child: dict[int, int] = {}
            for child_cluster in range(len(counts_curr)):
                mask = labels_curr == child_cluster
                parent_ids = np.unique(labels_prev[mask])
                if parent_ids.size != 1:
                    raise RuntimeError(
                        f"Cluster {child_cluster} at k={k} has multiple parents at k={k_prev}: {parent_ids}"
                    )
                parent_for_child[child_cluster] = int(parent_ids[0])

            parent_to_children: dict[int, list[int]] = {}
            for child_cluster, parent_cluster in parent_for_child.items():
                parent_to_children.setdefault(parent_cluster, []).append(child_cluster)

            for parent_cluster, children in parent_to_children.items():
                parent_color = cluster_color[(k_prev, parent_cluster)]
                if len(children) == 1:
                    cluster_color[(k, children[0])] = parent_color
                    continue

                children_sorted = sorted(children, key=lambda child: counts_curr[child], reverse=True)
                for idx, child_cluster in enumerate(children_sorted):
                    cluster_color[(k, child_cluster)] = parent_color if idx == 0 else next(color_gen)

            k_prev = k
            labels_prev = labels_curr

        dfs = []
        for k in k_vals:
            labels_k = labels_per_k[k]
            temp_df = pd.DataFrame({f"k{k}": labels_k})

            color_name_col = []
            color_rgb_col = []
            for lbl in labels_k:
                color_info = cluster_color[(k, int(lbl))]
                color_name_col.append(color_info["name"])
                color_rgb_col.append(color_info["val"])

            temp_df[f"k{k}_color"] = color_name_col
            temp_df[[f"k{k}_color_r", f"k{k}_color_g", f"k{k}_color_b"]] = np.array(color_rgb_col)
            dfs.append(temp_df)

        return pd.concat(dfs, axis=1)
