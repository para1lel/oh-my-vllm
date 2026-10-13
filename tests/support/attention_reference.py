import torch


def reference(query, cache, tables, lengths, first):
    q, pool = query.cpu().double(), cache.cpu().double()
    out = []
    for i, length in enumerate(lengths.cpu().tolist()):
        slots = torch.arange(first, length)
        pages = tables[i].cpu()[slots // 784].long()
        keys = pool[pages, 0, slots % 784].repeat_interleave(6, 1)
        values = pool[pages, 1, slots % 784].repeat_interleave(6, 1)
        scores = torch.einsum("hd,thd->ht", q[i], keys) / 16
        out.append(torch.einsum("ht,thd->hd", scores.softmax(-1), values))
    return torch.stack(out)
