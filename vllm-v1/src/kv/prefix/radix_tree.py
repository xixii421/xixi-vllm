""" 压缩前缀树（key=块哈希序列，value=物理块号序列） """

from __future__ import annotations

from collections.abc import Sequence as AbcSequence


def _prefix_len(a: AbcSequence[int], b: AbcSequence[int]) -> int:
    """ 两序列公共前缀长度。 """
    n = 0
    m = min(len(a), len(b))
    while n < m and a[n] == b[n]:
        n += 1
    return n


class _RadixNode:
    """ 树上一个节点：边上挂一段 key，节点上挂对应 value。 """

    __slots__ = ("children", "parent", "key", "value", "ref_count")

    def __init__(
        self,
        *,
        parent: _RadixNode | None = None,
        key: tuple[int, ...] = (),
        value: tuple[int, ...] = (),
        ref_count: int = 0,
    ) -> None:
        self.children: dict[int, _RadixNode] = {}
        self.parent = parent
        self.key = key
        self.value = value
        self.ref_count = ref_count


class RadixTree:
    """ insert / match_prefix；节点带简单引用计数。 """

    def __init__(self) -> None:
        self.root = _RadixNode(ref_count=1)
        self._block_to_node: dict[int, _RadixNode] = {}

    def match_prefix(self, key: list[int]) -> tuple[list[int], _RadixNode]:
        """ 沿树走最长匹配，返回已命中的 value（块号列表）与停留节点。 """
        if not key:
            return [], self.root
        node = self.root
        values: list[int] = []
        child_key = key[0]
        while key and child_key in node.children:
            child = node.children[child_key]
            pl = _prefix_len(child.key, key)
            if pl < len(child.key):
                new_node = self._split(child, pl)
                values.extend(new_node.value)
                return values, new_node
            values.extend(child.value)
            node = child
            key = key[pl:]
            child_key = key[0] if key else -1
        return values, node

    def insert(self, key: list[int], value: list[int]) -> _RadixNode:
        """ 插入一条 key→value，必要时分裂边；返回末端节点。 """
        if not key:
            return self.root
        node = self.root
        child_key = key[0]
        while key and child_key in node.children:
            child = node.children[child_key]
            pl = _prefix_len(child.key, key)
            key = key[pl:]
            value = value[pl:]
            child_key = key[0] if key else -1
            if pl < len(child.key):
                node = self._split(child, pl)
                break
            node = child
        if key:
            return self._add(node, tuple(key), tuple(value))
        return node

    def inc_ref(self, node: _RadixNode) -> None:
        """ 自 node 到根引用 +1。 """
        while node is not self.root:
            node.ref_count += 1
            node = node.parent or self.root

    def dec_ref(self, node: _RadixNode) -> None:
        """ 自 node 到根引用 -1。 """
        while node is not self.root:
            node.ref_count -= 1
            node = node.parent or self.root

    def _index(self, node: _RadixNode) -> None:
        """ 登记 value 中块号 → 节点。 """
        for bid in node.value:
            self._block_to_node[bid] = node

    def _unindex(self, node: _RadixNode) -> None:
        """ 取消该节点的块号登记。 """
        for bid in node.value:
            if self._block_to_node.get(bid) is node:
                del self._block_to_node[bid]

    def _add(
        self, parent: _RadixNode, key: tuple[int, ...], value: tuple[int, ...]
    ) -> _RadixNode:
        """ 在 parent 下挂一条新边。 """
        node = _RadixNode(parent=parent, key=key, value=value)
        parent.children[key[0]] = node
        self._index(node)
        return node

    def _split(self, node: _RadixNode, split_at: int) -> _RadixNode:
        """ 在边上 split_at 处拆成父边 + 子边。 """
        self._unindex(node)
        new_node = _RadixNode(
            parent=node.parent,
            key=node.key[:split_at],
            value=node.value[:split_at],
            ref_count=node.ref_count,
        )
        new_node.children = {node.key[split_at]: node}
        node.parent = new_node
        node.key = node.key[split_at:]
        node.value = node.value[split_at:]
        if new_node.parent is not None and new_node.key:
            new_node.parent.children[new_node.key[0]] = new_node
        self._index(new_node)
        self._index(node)
        return new_node
