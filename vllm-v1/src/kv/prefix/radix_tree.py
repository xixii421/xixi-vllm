""" 压缩前缀树 """

from __future__ import annotations

from collections.abc import Sequence as AbcSequence
# Sequence 抽象基类，代表有序可索引、可切片、可迭代的序列类型 因为自己定义了Sequence，所以重命名

def _prefix_len(a: AbcSequence[int], b: AbcSequence[int]) -> int:
    """ 两序列公共前缀长度 """
    n = 0
    m = min(len(a),len(b))
    while n < m and a[n] == b[n]:
        n+=1
    return n

class _RadixNode:
    """ 树上一个节点：边上挂一段 key，节点上挂对应 value """
    __slots__ = ("children", "parent", "key", "value", "ref_count")
    # Python优化，首先禁止实例生成字典_dict_ 同时禁止实例添加新属性
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
    """ insert / match_prefix；节点带简单引用计数 """
    def __init__(self) -> None:
        self.root = _RadixNode(ref_count=1)

    def _add(
            self, parent: _RadixNode, key: tuple[int,...], value: tuple[int,...],
    ) -> _RadixNode:
        """ 在 parent 下挂一条新边 """
        node = _RadixNode(parent=parent, key=key, value=value)
        parent.children[key[0]] = node
        # 取这条边 token 片段第一个 token，作为 children 字典的索引 key
        return node

    def _split(self, node: _RadixNode, split_at: int) -> _RadixNode:
        """ 在边上 split_at 处拆成父边 + 子边 """
        new_node = _RadixNode(
            parent=node.parent,
            key=node.key[:split_at],
            value=node.value[:split_at],
            ref_count=node.ref_count,
        ) # 中间新增节点信息更新
        new_node.children = {node.key[split_at]: node} # 新增节点指向旧节点
        node.parent = new_node
        node.key = node.key[split_at:]
        node.value = node.value[split_at:] # 旧阶段信息更新
        if new_node.parent is not None and new_node.key:
            # 如果新节点不是根节点或者key为空，更新之前父节点的子节点信息
            new_node.parent.children[new_node.key[0]] = new_node
        return new_node


    def insert(self, key: list[int], value: list[int]) -> _RadixNode:
        """ 插入一条 key→value，必要时分裂边；返回末端节点。 """
        if not key:
            return self.root
        node = self.root # 从根开始
        child_key = key[0] # 取第一个token
        while key and child_key in node.children:
            # 还有待插入token且当前节点存在以此token开头的子分支
            child = node.children[child_key]
            p1 = _prefix_len(child.key, key)
            key = key[p1:]
            value = value[p1:]
            child_key = key[0] if key else -1
            if p1 < len(child.key):
                node = self._split(child,p1)
                break
            node = child # 继续走
        if key: # 还有待插入token，但是没有字分支，则创建新节点
            return self._add(node,tuple(key),tuple(value))
        return node

    def match_prefix(self, key:list[int]) -> tuple[list[int], _RadixNode]:
        """ 沿树走最长匹配，返回已命中的 value（块号列表）与停留节点 """
        if not key:
            return [],self.root
        node = self.root
        values: list[int] = [] # 收集命中的可复用bid
        child_key = key[0] # 第一个token
        while key and child_key in node.children:
            # 剩下还有token，并且当前节点有以child_key开头的子分支，进入循环
            child = node.children[child_key]

            p1 = _prefix_len(child.key, key) # token匹配

            if p1 < len(child.key): # 边的token只匹配上一部分，需要分裂边
                new_node = self._split(child,p1)
                values.extend(new_node.value)
                return values,new_node
            # 这条边token匹配成功
            values.extend(child.value) # 保存物理页号
            node = child # 节点向下移动
            key = key[p1:] # 切掉已经完成匹配的部分，剩下的部分继续匹配
            child_key = key[0] if key else -1 # 更新剩下的第一个token，如果key空了赋值-1
        return values,node

    def inc_ref(self, node: _RadixNode) -> None:
        """ 自 node 到根引用 +1 """
        while node is not self.root:
            node.ref_count += 1
            node = node.parent or self.root

    def dec_ref(self, node: _RadixNode) -> None:
        """ 自 node 到根引用 -1 """
        while node is not self.root:
            node.ref_count -= 1
            node = node.parent or self.root