import torch


def generate_local_global_mask(reac, prod, Qlen, total_heads, local_heads):
    reac_rc = torch.zeros_like(reac.batch_mask)
    prod_rc = torch.zeros_like(prod.batch_mask)
    reac_rc[reac.batch_mask] = reac.is_rc | (~reac.is_prod)
    prod_rc[prod.batch_mask] = prod.is_rc

    rcx = torch.cat([reac_rc, prod_rc], dim=1)
    rcx = rcx.unsqueeze(1).unsqueeze(-1)
    # [bs, 1, klen, 1]

    global_mask = torch.ones_like(rcx)

    rcx = rcx.repeat(1, Qlen, 1, local_heads)
    global_mask = global_mask.repeat(1, Qlen, 1, total_heads - local_heads)
    return torch.logical_not(torch.cat([rcx, global_mask], dim=-1))
