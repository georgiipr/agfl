DEFAULTS = {
    'heads': 4, 'agfl_variant': 'polynomial', 'K': 2, 'top_k': 'scheduled',
    'top_k_stage': 'scores', 'top_k_ties': 'threshold',
    'top_k_schedule': {'smax': .2, 'smin': .8, 'alpha': 3.0},
    'graph_normalization': 'softmax', 'post_topk_renormalize': True,
    'coefficient_activation': 'identity', 'coefficient_init': 'zeros',
    'learnable_coefficients': True, 'projection': 'split_input', 'qkv_bias': False,
    'filter_projection': 'separate', 'score_scaling': 'temperature',
    'hop_normalization': 'feature',
}
