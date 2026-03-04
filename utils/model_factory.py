import torch

from model import JointModel, RAlignEncoder


def tie_reac_prod_params(encoder):
    for layer in encoder.layers:
        layer.prod_mpnn = layer.reac_mpnn
        layer.prod_mpnn_ln = layer.reac_mpnn_ln
        layer.prod_fusion_ln = layer.reac_fusion_ln
        if hasattr(layer, 'reac_ue') and hasattr(layer, 'prod_ue'):
            layer.prod_ue = layer.reac_ue
        if hasattr(layer, 'reac_edge_ln') and hasattr(layer, 'prod_edge_ln'):
            layer.prod_edge_ln = layer.reac_edge_ln


def build_joint_model(args, dropout: float):
    encoder = RAlignEncoder(
        n_layer=args.n_layer,
        emb_dim=args.dim,
        edge_dim=args.dim,
        heads=args.heads,
        dropout=dropout,
        negative_slope=args.negative_slope,
        update_last_edge=False,
        fusion_mode=args.fusion_mode
    )
    if args.share_reac_prod_encoder:
        tie_reac_prod_params(encoder)

    return JointModel(
        encoder=encoder,
        dim=args.dim,
        dropout=dropout,
        heads=args.heads
    )


def resolve_device(device_id: int):
    if torch.cuda.is_available() and device_id >= 0:
        return torch.device(f'cuda:{device_id}')
    return torch.device('cpu')
