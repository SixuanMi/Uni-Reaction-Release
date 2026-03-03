import torch
from .layers import DotMhAttn
from .utils import graph2batch
from utils.tensor_utils import (
    generate_square_subsequent_mask, generate_topk_mask,
    generate_local_global_mask
)


class RegressionModel(torch.nn.Module):
    def __init__(self, encoder, condition_encoder, dim, heads, dropout=0.1):
        super(RegressionModel, self).__init__()
        self.encoder = encoder
        self.condition_encoder = condition_encoder
        self.out_head = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, 1)
        )
        self.pool_keys = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.pooler = DotMhAttn(
            Qdim=dim, Kdim=dim, Vdim=dim, Odim=dim,
            emb_dim=dim, num_heads=heads, dropout=dropout
        )
        self.xln = torch.nn.LayerNorm(dim)

    def forward(self, reac_graph, prod_graph, conditions=None, cross_mask=None):
        if conditions is not None and self.condition_encoder is not None:
            condition_dict = self.condition_encoder(conditions)
        else:
            condition_dict = {}
        x_reac, x_prod, _, _ = self.encoder(
            reac_graph=reac_graph, reac_batched_condition=condition_dict,
            prod_graph=prod_graph, prod_batched_condition=condition_dict
        )

        x_reac = graph2batch(x_reac, reac_graph.batch_mask)
        x_prod = graph2batch(x_prod, prod_graph.batch_mask)
        memory = torch.cat([x_reac, x_prod], dim=1)
        memory_mask = [reac_graph.batch_mask, prod_graph.batch_mask]
        memory_mask = torch.logical_not(torch.cat(memory_mask, dim=1))

        pool_key = self.pool_keys.repeat(memory.shape[0], 1, 1)
        pooled_results, p_attn = self.pooler(
            query=pool_key, key=memory, value=memory,
            key_padding_mask=memory_mask, attn_mask=cross_mask
        )
        reaction_emb = self.xln(pooled_results.squeeze(dim=1))
        return self.out_head(reaction_emb)


class CNYieldModel(torch.nn.Module):
    def __init__(
        self, encoder, condition_encoder, dim, heads, dropout=0.1, out_dim=2
    ):
        super(CNYieldModel, self).__init__()
        self.encoder = encoder
        self.condition_encoder = condition_encoder
        self.out_head = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, out_dim)
        )
        self.pool_keys = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.pooler = DotMhAttn(
            Qdim=dim, Kdim=dim, Vdim=dim, Odim=dim,
            emb_dim=dim, num_heads=heads, dropout=dropout
        )
        self.xln = torch.nn.LayerNorm(dim)

    def forward(self, reac_graph, prod_graph, conditions, cross_mask=None):
        condition_dict = self.condition_encoder(conditions)
        x_reac, x_prod, _, _ = self.encoder(
            reac_graph=reac_graph, reac_batched_condition=condition_dict,
            prod_graph=prod_graph, prod_batched_condition=condition_dict
        )

        x_reac = graph2batch(x_reac, reac_graph.batch_mask)
        x_prod = graph2batch(x_prod, prod_graph.batch_mask)
        memory = torch.cat([x_reac, x_prod], dim=1)
        memory_mask = [reac_graph.batch_mask, prod_graph.batch_mask]
        memory_mask = torch.logical_not(torch.cat(memory_mask, dim=1))

        pool_key = self.pool_keys.repeat(memory.shape[0], 1, 1)
        pooled_results, p_attn = self.pooler(
            query=pool_key, key=memory, value=memory,
            key_padding_mask=memory_mask, attn_mask=cross_mask
        )
        reaction_emb = self.xln(pooled_results.squeeze(dim=1))
        return self.out_head(reaction_emb)


class USPTOConditionModel(torch.nn.Module):
    def __init__(self, encoder, decoder, pe, n_words, dim):
        super(USPTOConditionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pe = pe
        self.word_emb = torch.nn.Embedding(n_words, dim)
        self.cat_emb = torch.nn.Parameter(torch.randn(dim))
        self.sov_emb = torch.nn.Parameter(torch.randn(dim))
        self.reg_emb = torch.nn.Parameter(torch.randn(dim))
        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, n_words)
        )

    def encode(self, reac_graph, prod_graph):
        x_reac, x_prod, _, _ = self.encoder(
            reac_graph=reac_graph, prod_graph=prod_graph
        )
        x_reac = graph2batch(x_reac, reac_graph.batch_mask)
        x_prod = graph2batch(x_prod, prod_graph.batch_mask)
        memory = torch.cat([x_reac, x_prod], dim=1)
        memory_mask = [reac_graph.batch_mask, prod_graph.batch_mask]
        memory_mask = torch.logical_not(torch.cat(memory_mask, dim=1))
        return memory, memory_mask

    def decode_a_step(
        self, memory, seq, memory_key_padding_mask=None, cross_mask=None,
        tgt_mask=None, tgt_key_padding_mask=None
    ):
        type_emb = torch.stack([
            torch.zeros_like(self.cat_emb), self.cat_emb,
            self.sov_emb, self.sov_emb, self.reg_emb
        ], dim=0)
        seq_emb = self.pe(self.word_emb(seq) + type_emb[:seq.shape[-1]])
        seq = self.decoder(
            tgt=seq_emb, memory=memory, tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask, memory_mask=cross_mask
        )
        return self.out_layer(seq)

    def forward(
        self, reac_graph, prod_graph, tgt, tgt_mask=None,
        tgt_key_padding_mask=None, cross_mask=None
    ):
        memory, memory_pad = self.encode(reac_graph, prod_graph)
        assert tgt.shape[1] <= 5, 'Invalid Format for prediction'
        result = self.decode_a_step(
            memory, tgt, tgt_mask=tgt_mask, cross_mask=cross_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_pad
        )

        return result

    def beam_search(
        self, reac_graph, prod_graph, res, total_heads=None,
        local_heads=0, beam_size=10
    ):
        def sq_ft(x):
            assert x.dim() >= 2, 'no enough dim to squeeze'
            return x.reshape((-1, *x.shape[2:]))

        memory, memory_padding = self.encode(reac_graph, prod_graph)
        (bs, ml), device = memory.shape[:2], memory.device
        log_logits = torch.zeros(bs).to(device)
        belong = torch.arange(0, bs).to(device)

        unit_cross_mask = generate_local_global_mask(
            reac_graph, prod_graph, 1, total_heads, local_heads
        ) if local_heads > 0 else None

        for i in range(5):
            cross_mask = None if unit_cross_mask is None else\
                unit_cross_mask[belong].repeat(1, i + 1, 1, 1)

            mem_cand, mem_pad_cand, seq_cand = [], [], []
            log_cand, bel_cand = [], []
            diag_mask = generate_square_subsequent_mask(i + 1, device)
            rc_out = torch.log_softmax(self.decode_a_step(
                memory=memory,  cross_mask=cross_mask, tgt_mask=diag_mask,
                seq=res, memory_key_padding_mask=memory_padding
            )[:, -1], dim=-1)

            dup = min(rc_out.shape[-1], beam_size)

            memory = memory[:, None].repeat(1, dup, 1, 1)
            # [alive_n, beam, ml, dim]
            memory_padding = memory_padding[:, None].repeat(1, dup, 1)
            # [alive_n, dup, ml]
            res = res[:, None].repeat(1, dup, 1)
            # [alive_n, dup, seq_len]
            belong = belong[:, None].repeat(1, dup)
            # [alive_n, dup]

            top_res = torch.topk(rc_out, k=dup, dim=-1, largest=True)
            log_logits = log_logits[:, None] + top_res.values
            res = torch.cat([res, top_res.indices.unsqueeze(dim=-1)], dim=-1)

            mem_cand.append(sq_ft(memory))
            mem_pad_cand.append(sq_ft(memory_padding))
            seq_cand.append(sq_ft(res))
            log_cand.append(sq_ft(log_logits))
            bel_cand.append(sq_ft(belong))

            memory = torch.cat(mem_cand, dim=0)
            memory_padding = torch.cat(mem_pad_cand, dim=0)
            res = torch.cat(seq_cand, dim=0)
            log_logits = torch.cat(log_cand, dim=0)
            belong = torch.cat(bel_cand, dim=0)

            sort_out = torch.sort(log_logits, descending=True)
            log_logits = sort_out.values
            memory = memory[sort_out.indices]
            res = res[sort_out.indices]
            memory_padding = memory_padding[sort_out.indices]
            belong = belong[sort_out.indices]

            topk_mask = generate_topk_mask(belong, bs, beam_size)
            log_logits = log_logits[topk_mask]
            res = res[topk_mask]
            belong = belong[topk_mask]
            memory = memory[topk_mask]
            memory_padding = memory_padding[topk_mask]

        return res, log_logits, belong


class USPTO500MTModel(torch.nn.Module):
    def __init__(self, encoder, decoder, pe, n_words, dim):
        super(USPTO500MTModel, self).__init__()
        self.word_emb = torch.nn.Embedding(n_words, dim)
        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, n_words)
        )
        self.encoder = encoder
        self.decoder = decoder
        self.pe = pe

    def encode(self, reac_graph, prod_graph):
        x_reac, x_prod, _, _ = self.encoder(
            reac_graph=reac_graph, prod_graph=prod_graph
        )
        x_reac = graph2batch(x_reac, reac_graph.batch_mask)
        x_prod = graph2batch(x_prod, prod_graph.batch_mask)
        memory = torch.cat([x_reac, x_prod], dim=1)
        memory_mask = [reac_graph.batch_mask, prod_graph.batch_mask]
        memory_mask = torch.logical_not(torch.cat(memory_mask, dim=1))
        return memory, memory_mask

    def decode_a_step(
        self, memory, seq, memory_key_padding_mask=None, cross_mask=None,
        tgt_mask=None, tgt_key_padding_mask=None
    ):
        seq_emb = self.pe(self.word_emb(seq))
        seq = self.decoder(
            tgt=seq_emb, memory=memory, tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask, memory_mask=cross_mask
        )
        return self.out_layer(seq)

    def forward(
        self, reac_graph, prod_graph, tgt, tgt_mask=None,
        tgt_key_padding_mask=None, cross_mask=None
    ):
        memory, memory_pad = self.encode(reac_graph, prod_graph)
        result = self.decode_a_step(
            memory, tgt, tgt_mask=tgt_mask, cross_mask=cross_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_pad
        )

        return result

    def beam_search(
        self, reac_graph, prod_graph, res, padding_idx, end_idx,
        beam=10, max_len=400, total_heads=None, local_heads=0,
        left_parenthesis_idx=-1, right_parenthesis_idx=-1
    ):
        def sq_ft(x):
            assert x.dim() >= 2, 'no enough dim to squeeze'
            return x.reshape((-1, *x.shape[2:]))
        memory, memory_padding = self.encode(reac_graph, prod_graph)
        (bs, ml), device = memory.shape[:2], memory.device
        log_logits = torch.zeros(bs).to(device)
        belong = torch.arange(0, bs).to(device)

        unit_cross_mask = generate_local_global_mask(
            reac_graph, prod_graph, 1, total_heads, local_heads
        ) if local_heads > 0 else None

        if (
            (left_parenthesis_idx != -1 and right_parenthesis_idx == -1) or
            (left_parenthesis_idx == -1 and right_parenthesis_idx != -1)
        ):
            raise ValueError(
                "The idx of left and right parenthesis should " +
                "be given at the same time"
            )

        n_close = torch.zeros(bs, dtype=torch.long).to(device)
        alive = torch.ones(bs, dtype=bool).to(device)
        begin_len = res.shape[1]
        for i in range(max_len):
            dead = torch.logical_not(alive)
            n_alive, n_dead = alive.sum().item(), dead.sum().item()
            if n_dead > 0:
                to_pad = torch.ones(n_dead, dtype=torch.long)
                to_pad = (to_pad.to(device) * padding_idx).reshape(-1, 1)
                seq_cand = [torch.cat([res[dead], to_pad], dim=-1)]
                bel_cand = [belong[dead]]
                log_cand = [log_logits[dead]]
                mem_cand = [memory[dead]]
                mem_pad_cand = [memory_padding[dead]]
                close_cand = [n_close[dead]]
                alive_cand = [torch.zeros(n_dead, dtype=bool).to(device)]
            else:
                seq_cand, bel_cand, log_cand, alive_cand = [], [], [], []
                close_cand, mem_cand, mem_pad_cand = [], [], []

            memory, memory_padding = memory[alive], memory_padding[alive]
            n_close, res = n_close[alive], res[alive]
            log_logits, belong = log_logits[alive], belong[alive]

            cross_mask = None if unit_cross_mask is None else\
                unit_cross_mask[belong].repeat(1, i + begin_len, 1, 1)

            diag_mask = generate_square_subsequent_mask(res.shape[1], device)
            rc_out = torch.log_softmax(self.decode_a_step(
                memory=memory,  cross_mask=cross_mask, tgt_mask=diag_mask,
                seq=res, memory_key_padding_mask=memory_padding
            )[:, -1], dim=-1)

            dup = min(rc_out.shape[-1], beam)

            memory = memory[:, None].repeat(1, dup, 1, 1)
            # [alive_n, beam, ml, dim]
            memory_padding = memory_padding[:, None].repeat(1, dup, 1)
            # [alive_n, dup, ml]
            res = res[:, None].repeat(1, dup, 1)
            # [alive_n, dup, seq_len]
            belong = belong[:, None].repeat(1, dup)
            # [alive_n, dup]
            n_close = n_close[:, None].repeat(1, dup)
            # [alive_n, dup]

            top_res = torch.topk(rc_out, k=dup, dim=-1, largest=True)
            log_logits = log_logits[:, None] + top_res.values
            res = torch.cat([res, top_res.indices.unsqueeze(dim=-1)], dim=-1)
            alivex = top_res.indices != end_idx
            is_fst = (top_res.indices == left_parenthesis_idx).long()
            is_sec = (top_res.indices == right_parenthesis_idx).long()
            n_close = n_close + is_fst - is_sec

            mem_cand.append(sq_ft(memory))
            mem_pad_cand.append(sq_ft(memory_padding))
            seq_cand.append(sq_ft(res))
            log_cand.append(sq_ft(log_logits))
            close_cand.append(sq_ft(n_close))
            bel_cand.append(sq_ft(belong))
            alive_cand.append(sq_ft(alivex))

            memory = torch.cat(mem_cand, dim=0)
            memory_padding = torch.cat(mem_pad_cand, dim=0)
            res = torch.cat(seq_cand, dim=0)
            log_logits = torch.cat(log_cand, dim=0)
            belong = torch.cat(bel_cand, dim=0)
            alive = torch.cat(alive_cand, dim=0)
            n_close = torch.cat(close_cand, dim=0)
            illegal = (n_close < 0) | ((~alive) & (n_close != 0))
            log_logits[illegal] = float('-inf')

            sort_out = torch.sort(log_logits, descending=True)
            log_logits = sort_out.values
            memory = memory[sort_out.indices]
            res = res[sort_out.indices]
            memory_padding = memory_padding[sort_out.indices]
            belong = belong[sort_out.indices]
            alive = alive[sort_out.indices]
            n_close = n_close[sort_out.indices]

            topk_mask = generate_topk_mask(belong, bs, beam)
            log_logits = log_logits[topk_mask]
            res = res[topk_mask]
            belong = belong[topk_mask]
            memory = memory[topk_mask]
            memory_padding = memory_padding[topk_mask]
            n_close = n_close[topk_mask]
            alive = alive[topk_mask]

            if not torch.any(alive).item():
                break

        return res, log_logits, belong

class JointModel(torch.nn.Module):
    def __init__(self, encoder, condition_encoder, dim, heads, dropout=0.1, cls_out_dim=2):
        super(JointModel, self).__init__()
        # 共享编码器
        self.encoder = encoder
        self.condition_encoder = condition_encoder
        
        # 分类任务头（例如：反应类型分类）
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(dim * 2, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, cls_out_dim)  # 2类分类
        )
        
        # 回归任务头（例如：产率预测）
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(dim * 3, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, 1),
            torch.nn.Softplus()  # 保证能垒输出非负，训练/评估语义一致
        )
        
        # 反应物/产物共享同一个池化query，减少额外方向偏置
        self.pool_keys = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.pooler = DotMhAttn(
            Qdim=dim, Kdim=dim, Vdim=dim, Odim=dim,
            emb_dim=dim, num_heads=heads, dropout=dropout
        )
        self.xln = torch.nn.LayerNorm(dim)

    def forward(self, reac_graph, prod_graph, conditions=None, cross_mask=None):
        # 共享特征提取
        if conditions is not None and self.condition_encoder is not None:
            condition_dict = self.condition_encoder(conditions)
        else:
            condition_dict = {}
        x_reac, x_prod, _, _ = self.encoder(
            reac_graph=reac_graph, reac_batched_condition=condition_dict,
            prod_graph=prod_graph, prod_batched_condition=condition_dict
        )
        
        # 分别池化反应物和产物，保留方向信息供后续组合
        x_reac = graph2batch(x_reac, reac_graph.batch_mask)
        x_prod = graph2batch(x_prod, prod_graph.batch_mask)
        reac_mask = torch.logical_not(reac_graph.batch_mask)
        prod_mask = torch.logical_not(prod_graph.batch_mask)

        shared_query = self.pool_keys.repeat(x_reac.shape[0], 1, 1)
        reac_query = shared_query
        prod_query = shared_query
        reac_cross_mask, prod_cross_mask = None, None

        if cross_mask is not None and cross_mask.ndim == 4:
            mask_source = cross_mask[:, :1] if cross_mask.shape[1] != 1 else cross_mask
            total_len = x_reac.shape[1] + x_prod.shape[1]
            if mask_source.shape[2] != total_len:
                raise ValueError(
                    f'cross_mask key dim {mask_source.shape[2]} does not match '
                    f'combined sequence length {total_len}'
                )
            reac_cross_mask = mask_source[:, :, :x_reac.shape[1], :]
            prod_cross_mask = mask_source[:, :, x_reac.shape[1]:, :]

        r_pool, _ = self.pooler(
            query=reac_query, key=x_reac, value=x_reac,
            key_padding_mask=reac_mask, attn_mask=reac_cross_mask
        )
        p_pool, _ = self.pooler(
            query=prod_query, key=x_prod, value=x_prod,
            key_padding_mask=prod_mask, attn_mask=prod_cross_mask
        )
        r_pool = self.xln(r_pool.squeeze(dim=1))
        p_pool = self.xln(p_pool.squeeze(dim=1))
        d_pool = p_pool - r_pool

        cls_feat = torch.cat([r_pool + p_pool, torch.abs(d_pool)], dim=-1)
        reg_feat = torch.cat([r_pool, p_pool, d_pool], dim=-1)
        
        # 双任务输出
        cls_out = self.cls_head(cls_feat)  # 分类输出：交换R/P保持不变
        reg_out = self.reg_head(reg_feat)  # 回归输出：显式保留方向信息
        return cls_out, reg_out
