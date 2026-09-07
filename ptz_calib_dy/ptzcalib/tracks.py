"""Port of ``src/core/tracks.{h,cc}`` (and ``flat_pair_map.h``).

``Track``  -> ``dict[image_id, feature_id]`` kept sorted by ``image_id``
``Tracks`` -> ``dict[track_id, Track]``     kept sorted by ``track_id``
(mirrors ``std::map`` iteration order, which the optimizer relies on).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from .types import MatchesInfo
from .union_find import UnionFind

Track = Dict[int, int]
Tracks = Dict[int, Track]

_INT_MAX = 2147483647
_FEAT_SHIFT = 1 << 31  # (image_id, feature_id) -> single sortable key


class TracksBuilder:
    """``class TracksBuilder``"""

    def __init__(self) -> None:
        # flat_pair_map<IndexedFeaturePair, int>: sorted (image_id, feature_id) keys
        self._keys: np.ndarray = np.zeros((0,), dtype=np.int64)
        self._img_ids: np.ndarray = np.zeros((0,), dtype=np.int64)
        self._feat_ids: np.ndarray = np.zeros((0,), dtype=np.int64)
        self.uf_tree_ = UnionFind()

    @property
    def map_node_to_index_size(self) -> int:
        return int(self._keys.size)

    def build(self, matches_info: Sequence[MatchesInfo]) -> None:
        # 1./2. collect every (imageIndex, featureIndex) node and give it a unique
        # index; std::set gives ascending (image_id, feature_id) order.
        img_parts: List[np.ndarray] = []
        feat_parts: List[np.ndarray] = []
        for mi in matches_info:
            if mi is None or mi.matches.shape[0] == 0:
                continue
            m = mi.matches
            img_parts.append(np.full(m.shape[0], mi.src_img_idx, dtype=np.int64))
            feat_parts.append(m[:, 0].astype(np.int64))
            img_parts.append(np.full(m.shape[0], mi.dst_img_idx, dtype=np.int64))
            feat_parts.append(m[:, 1].astype(np.int64))

        if not img_parts:
            self._keys = np.zeros((0,), dtype=np.int64)
            self._img_ids = self._keys
            self._feat_ids = self._keys
            self.uf_tree_.init_sets(0)
            return

        imgs = np.concatenate(img_parts)
        feats = np.concatenate(feat_parts)
        keys = imgs * _FEAT_SHIFT + feats
        self._keys = np.unique(keys)
        self._img_ids = self._keys // _FEAT_SHIFT
        self._feat_ids = self._keys % _FEAT_SHIFT

        # 3. init the UF forest
        self.uf_tree_.init_sets(int(self._keys.size))

        # 4. union of the matched features
        union = self.uf_tree_.union
        for mi in matches_info:
            if mi is None or mi.matches.shape[0] == 0:
                continue
            m = mi.matches
            idx_i = np.searchsorted(self._keys, mi.src_img_idx * _FEAT_SHIFT + m[:, 0].astype(np.int64))
            idx_j = np.searchsorted(self._keys, mi.dst_img_idx * _FEAT_SHIFT + m[:, 1].astype(np.int64))
            for a, b in zip(idx_i.tolist(), idx_j.tolist()):
                union(a, b)

    def filter(self, min_track_length: int = 2) -> None:
        """``TracksBuilder::Filter``: drop tracks with an image collision or too short."""
        uf = self.uf_tree_
        n = self.map_node_to_index_size
        img_ids = self._img_ids

        tracks: Dict[int, Set[int]] = {}
        problematic: Set[int] = set()

        find = uf.find
        for k in range(n):
            track_id = find(k)
            img_id = int(img_ids[k])
            bucket = tracks.get(track_id)
            if bucket is None:
                tracks[track_id] = {img_id}
            elif img_id in bucket:
                problematic.add(track_id)  # an image can only be listed once
            else:
                bucket.add(img_id)

        for track_id, imgs in tracks.items():
            if len(imgs) < min_track_length:
                problematic.add(track_id)

        parent = uf.m_cc_parent
        size = uf.m_cc_size
        for i in range(len(parent)):
            root_index = parent[i]
            if root_index in problematic:
                size[root_index] = 1
                parent[i] = _INT_MAX

    def export_to_stl(self) -> Tracks:
        """``TracksBuilder::ExportToSTL``"""
        uf = self.uf_tree_
        parent = uf.m_cc_parent
        size = uf.m_cc_size
        img_ids = self._img_ids
        feat_ids = self._feat_ids

        tracks: Tracks = {}
        for k in range(self.map_node_to_index_size):
            track_id = parent[k]
            if track_id != _INT_MAX and size[track_id] > 1:
                track = tracks.get(track_id)
                if track is None:
                    track = {}
                    tracks[track_id] = track
                img_id = int(img_ids[k])
                if img_id not in track:  # std::map::insert keeps the first value
                    track[img_id] = int(feat_ids[k])

        # mirror std::map ordering (track ids ascending, image ids ascending)
        return {tid: dict(sorted(tracks[tid].items())) for tid in sorted(tracks)}

    def nb_tracks(self) -> int:
        parent_id = set(self.uf_tree_.m_cc_parent)
        parent_id.discard(_INT_MAX)
        return len(parent_id)


def length(tracks: Tracks) -> Tuple[int, int, int]:
    """``Length``: returns ``(total_length, max_length, min_length)``."""
    total_length = 0
    max_length = 0
    min_length = _INT_MAX
    for track in tracks.values():
        n = len(track)
        total_length += n
        max_length = max(max_length, n)
        min_length = min(min_length, n)
    return total_length, max_length, min_length


def _is_connected(img_id: int, connected_img_sets: List[Set[int]]) -> int:
    for i, s in enumerate(connected_img_sets):
        if img_id in s:
            return i
    return -1


def find_max_covisible(tracks: Tracks, num_images: int) -> Set[int]:
    """``FindMaxCoVisible``: largest co-visible image set."""
    connected_img_sets: List[Set[int]] = []

    for track in tracks.values():
        is_not_connect = True
        connected_idx_all: Set[int] = set()
        for img_id in track:
            connected_idx = _is_connected(img_id, connected_img_sets)
            if connected_idx != -1:
                is_not_connect = False
                connected_idx_all.add(connected_idx)

        co_visible_new = set(track.keys())

        if is_not_connect:
            connected_img_sets.append(co_visible_new)
        else:
            for idx in sorted(connected_idx_all, reverse=True):
                co_visible_new |= connected_img_sets[idx]
            connected_img_sets.append(co_visible_new)
            for idx in sorted(connected_idx_all, reverse=True):
                del connected_img_sets[idx]

    max_connect_imgs: Set[int] = set()
    max_num = 0
    for co_visible in connected_img_sets:
        if len(co_visible) > max_num:
            max_num = len(co_visible)
            max_connect_imgs = co_visible
    return max_connect_imgs


def save_tracks(tracks: Tracks, img_names: Sequence[str], outpath: str) -> None:
    """``SaveTracks``"""
    try:
        with open(outpath, "w", encoding="utf-8") as fout:
            for track in tracks.values():
                parts = []
                for img_id, feature_id in track.items():
                    parts.append(f"{img_names[img_id]} {feature_id}")
                fout.write(" ".join(parts) + "\n")
    except OSError:
        import sys

        print(f"Cannot write file to {outpath}", file=sys.stderr)
