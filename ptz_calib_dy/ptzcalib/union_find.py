"""Port of ``src/core/union_find.h``.

``Find`` is written iteratively (the C++ version recurses); the resulting parent
array after full path compression is identical.
"""

from __future__ import annotations

from typing import List


class UnionFind:
    __slots__ = ("m_cc_parent", "m_cc_rank", "m_cc_size")

    def __init__(self) -> None:
        self.m_cc_parent: List[int] = []
        self.m_cc_rank: List[int] = []
        self.m_cc_size: List[int] = []

    def init_sets(self, num_cc: int) -> None:
        self.m_cc_size = [1] * num_cc
        self.m_cc_parent = list(range(num_cc))
        self.m_cc_rank = [0] * num_cc

    def get_num_nodes(self) -> int:
        return len(self.m_cc_size)

    def find(self, i: int) -> int:
        parent = self.m_cc_parent
        if i < 0 or i >= len(parent):
            raise IndexError("Index out of range")

        root = i
        while parent[root] != root:
            root = parent[root]
        # path compression (same fixed point as the recursive version)
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    def union(self, i: int, j: int) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i == root_j:
            return

        rank = self.m_cc_rank
        size = self.m_cc_size
        parent = self.m_cc_parent
        if rank[root_i] < rank[root_j]:
            parent[root_i] = root_j
            size[root_j] += size[root_i]
        else:
            parent[root_j] = root_i
            size[root_i] += size[root_j]
            if rank[root_i] == rank[root_j]:
                rank[root_i] += 1

    def connected(self, i: int, j: int) -> bool:
        return self.find(i) == self.find(j)

    def size(self, i: int) -> int:
        return self.m_cc_size[self.find(i)]
