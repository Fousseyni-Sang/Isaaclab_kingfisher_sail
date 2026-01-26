import torch

def sample_feasible_pairs_batch(env_ids, params, device="cpu", batch_factor=10):
    """
    env_ids: LongTensor of env indices that need new (V, omega) targets
    params dict keys:
        'V_min', 'V_max',
        'omega_max',
        'V_min_nonzero',
        'R_min'   <-- only constraint used besides bounds!
    """

    num = len(env_ids)
    if num == 0:
        return None

    # Oversample to reduce rejection rate
    N = num * batch_factor

    # Sample candidates
    V = torch.empty(N, device=device).uniform_(params['V_min'], params['V_max'])
    omega = torch.empty(N, device=device).uniform_(-params['omega_max'], params['omega_max'])

    # ---- FEASIBILITY CHECKS ----
    
    eps = 1e-6
    tiny_omega = omega.abs() < eps


    # 2) Minimum turn radius: |V/omega| >= R_min, except when omega ≈ 0
    mask_radius = torch.ones(N, dtype=torch.bool, device=device)
    nonzero_omega = ~tiny_omega
    mask_radius[nonzero_omega] = (
        (V[nonzero_omega].abs() / omega[nonzero_omega].abs()) >= params['R_min']
    )
    # if omega == 0 → straight line → infinite radius → allowed

    feasible = mask_radius

    # Extract feasible samples
    V_f = V[feasible]
    w_f = omega[feasible]

    # Not enough? Recursively fill the remainder
    if len(V_f) < num:
        extra_needed = num - len(V_f)
        extra_pairs = sample_feasible_pairs_batch(
            env_ids=torch.arange(extra_needed, device=device),
            params=params,
            device=device,
            batch_factor=batch_factor
        )
        return torch.cat(
            [torch.stack([V_f, w_f], dim=-1), extra_pairs],
            dim=0
        )[:num]

    # Enough feasible ones
    V_res = V_f[:num]
    w_res = w_f[:num]

    return torch.stack([V_res, w_res], dim=-1)
