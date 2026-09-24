DEFAULTS = {
    'heads': 4, 'agfl_variant': 'polynomial', 'K': 2, 'top_k': 'scheduled',
    'top_k_stage': 'scores', 'top_k_ties': 'threshold',
    'top_k_schedule': {'smax': .2, 'smin': .8, 'alpha': 3.0},
    'graph_normalization': 'softmax', 'post_topk_renormalize': True,
    'coefficient_activation': 'identity', 'coefficient_init': 'zeros',
    # token_contrast is an explicit attention-only extension; old defaults stay static.
    'coefficient_conditioning': 'static', 'coefficient_conditioning_scale': 0.5,
    'learnable_coefficients': True, 'projection': 'split_input', 'qkv_bias': False,
    'filter_projection': 'separate', 'filter_projection_init': 'pytorch',
    'score_scaling': 'temperature',
    'temperature_init': 1.0,
    'hop_normalization': 'feature',
}
