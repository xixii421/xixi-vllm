from __future__ import annotations
import torch

class Sampler:
    def __call__(self,
        logits: torch.Tensor,
        temperatures: torch.Tensor | None = None,
        top_ps: torch.Tensor | None = None,
        top_ks: torch.Tensor | None = None,
    )->torch.Tensor:
        # logits hidden_states经过lm_head 输出投影层的处理的到每个词的logit分数
        # logits = lm_head(hidden_states) logits.shape = [B,vocab_size]
        # 代表每个候选token的原始分数，因为还没有做softmax，所以还没有概率含义
        if temperatures is None and top_ps is None and top_ks is None:
            return torch.argmax(logits, dim=-1)
        # temperatures top-k top-p 都不做，直接返回概率最大的token下标，贪心采样

        logits = logits.float()
        if temperatures is not None:
            temperatures = torch.where(temperatures <= 0, 1.0, temperatures)
            logits = logits / temperatures.unsqueeze(1)
        logits = _top_k_top_p(logits, top_ps, top_ks)
        probs = torch.softmax(logits, dim=-1) # 转为概率分布
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
        # multinomial 按概率分布做多项式随机采样 按照概率，抽取nums_samples个token下标
        # 最后返回结果 shape=[B],每个元素就是每个batch样本采样得到的下一个token_id

def _top_k_top_p(
    logits: torch.Tensor,
    p: torch.Tensor | None = None,
    k: torch.Tensor | None = None,
)->torch.Tensor:
    # 这里p,k输入是张量是因为需要确保每个batch用不同的pk参数 p.shape = [B]
    if p is None and k is None:
        return logits # top-k top-p都不开启，不过滤直接返回
    vocab = logits.size(-1) # 取词表大小
    sorted_logits,sorted_idx = logits.sort(dim=-1,descending=False)
    # 升序排序，排序结果返回 sorted_logits 对应排序后下标存到 sorted_idx

    if k is not None:
        k = torch.where(k<=0,vocab,k)
        # 如果k<=0 直接复制为vocab，相当于不做top-k，否则保留
        cutoff = sorted_logits.gather(1,(vocab-k.to(torch.long)).unsqueeze(1))
        # 1 vocab-k 得到升序数组里top-k的起始下标，此时shape变为[B] .to(torch.long) 转换为整数索引
        # 2 unsqueeze(1) 扩维，把shape变为[B,1]
        # 3 gather 在输入的维度中，按照输入给出的下标取出对应元素
        # 4 cutoff 最后得到的结果是每个batch行分割点位置的logit分数，此时cutoff.shape=[B,1]
        sorted_logits = sorted_logits.masked_fill(sorted_logits<cutoff,-float("inf"))
        # masked_fill 掩码填充，掩码为Ture的位置，填充负无穷

    if p is not None:
        p = torch.where(p <= 0, 1.0, p)
        probs = sorted_logits.softmax(dim=-1)
        # 1 sorted_logits.shape=[B,V]
        # 2 softmax，把排序后的logits转成概率分布
        mask = probs.cumsum(dim=-1) <= (1 - p).unsqueeze(1)
        # cumsum 沿着词表维度做累计求和，累计概率小于1-p的位置值True，后面要淘汰
        mask[:, -1] = False
        # 强制把最后一列，也就是概率最大的token置False，兜底保护
        sorted_logits = sorted_logits.masked_fill(mask, -float("inf"))
    return torch.empty_like(sorted_logits).scatter_(1, sorted_idx, sorted_logits)
    # 1 经过top-k top-p 过滤，但是顺序是升序，需要按照之前的下标还原
    # 2 empty_like创建一个和sorted_logits的shape,dtype一模一样的张量 shape=[B,V]
    # 3 scatter_ 按照第二维度写入，写入方式为 out[i,sorted_idx[i,j]]=src[i,j]
