import torch


SHIELD_PARAMS = {
    "a_low": -3.0,
    "a_high": 2.0,
    "a_brake_hard": -2.2,
    "a_brake_moderate": -1.2,
    "a_brake_light": -0.8,
    "a_idle": 0.0,
    "a_slight_accel": -0.1,
    "a_moderate_accel": 0.2,
    "ttc_critical": 1.0,
    "ttc_warning": 1.8,
    "ttc_brake_critical": -3.0,
    "ttc_brake_warning": -2.4,
    "d_safe_pred_ratio": 0.98,
    "d_safe_relv_fast": -1.2,
    "d_safe_relv_moderate": -0.3,
    "d_safe_ratio_fast": 1.12,
    "d_safe_ratio_moderate": 1.05,
    "gap_upper_ratio_1": 1.40,
    "gap_upper_ratio_2": 1.35,
    "gap_upper_ratio_3": 1.60,
    "gap_upper_abs_1": 35.0,
    "gap_upper_abs_2": 30.0,
    "gap_upper_abs_3": 45.0,
    "relv_proactive_1": 0.3,
    "relv_proactive_2": 0.5,
    "relv_proactive_3": 1.0,
}


def _compute_d_safe_vectorized(v_ego, v_lead, sigma_mean):
    delta_v = torch.relu(v_ego - v_lead)
    return 5.0 + 1.2 * v_ego + (delta_v * delta_v) / (2.0 * 4.0 + 1e-6) + 0.9 * sigma_mean


def shield_action_from_obs_vectorized(obs, act, dt_pred=1.0):
    p = SHIELD_PARAMS
    a_low, a_high = p["a_low"], p["a_high"]

    v_ego = obs[:, 0:1]
    d_gap = obs[:, 2:3]
    v_lead = obs[:, 3:4]
    sigma_mean = obs[:, 5:6]

    d_safe = _compute_d_safe_vectorized(v_ego, v_lead, sigma_mean)
    rel_speed = v_lead - v_ego
    a = torch.clamp(act, a_low, a_high)

    closing_speed = torch.relu(-rel_speed)
    ttc = torch.where(closing_speed > 1e-6, d_gap / (closing_speed + 1e-6), torch.full_like(d_gap, 1e6))

    pred_gap = d_gap - closing_speed * dt_pred
    a = torch.where(pred_gap < p["d_safe_pred_ratio"] * d_safe, torch.minimum(a, torch.full_like(a, p["a_brake_hard"])), a)
    a = torch.where(d_gap < d_safe, torch.minimum(a, torch.full_like(a, p["a_brake_hard"])), a)
    a = torch.where((rel_speed < p["d_safe_relv_fast"]) & (d_gap < d_safe * p["d_safe_ratio_fast"]), torch.minimum(a, torch.full_like(a, p["a_brake_moderate"])), a)
    a = torch.where((rel_speed < p["d_safe_relv_moderate"]) & (d_gap < d_safe * p["d_safe_ratio_moderate"]), torch.minimum(a, torch.full_like(a, p["a_brake_light"])), a)

    a = torch.where(ttc < p["ttc_critical"], torch.minimum(a, torch.full_like(a, p["ttc_brake_critical"])), a)
    a = torch.where((ttc >= p["ttc_critical"]) & (ttc < p["ttc_warning"]), torch.minimum(a, torch.full_like(a, p["ttc_brake_warning"])), a)

    a = torch.where((d_gap > torch.maximum(p["gap_upper_ratio_1"] * d_safe, torch.full_like(d_safe, p["gap_upper_abs_1"]))) & (rel_speed > p["relv_proactive_1"]), torch.maximum(a, torch.full_like(a, p["a_idle"])), a)
    a = torch.where((d_gap > torch.maximum(p["gap_upper_ratio_2"] * d_safe, torch.full_like(d_safe, p["gap_upper_abs_2"]))) & (rel_speed > p["relv_proactive_2"]), torch.maximum(a, torch.full_like(a, p["a_slight_accel"])), a)
    a = torch.where((d_gap > torch.maximum(p["gap_upper_ratio_3"] * d_safe, torch.full_like(d_safe, p["gap_upper_abs_3"]))) & (rel_speed > p["relv_proactive_3"]), torch.maximum(a, torch.full_like(a, p["a_moderate_accel"])), a)

    return torch.clamp(a, a_low, a_high)


def shield_action_scalar(acc_cmd, d_gap, d_safe, rel_speed, dt_pred=1.0):
    p = SHIELD_PARAMS
    a_low, a_high = p["a_low"], p["a_high"]
    a = float(max(a_low, min(acc_cmd, a_high)))

    closing_speed = max(-rel_speed, 0.0)
    ttc = d_gap / (closing_speed + 1e-6) if closing_speed > 1e-6 else float("inf")

    dt_p = max(1e-3, min(float(dt_pred), 5.0))
    pred_gap = d_gap - closing_speed * dt_p

    if pred_gap < p["d_safe_pred_ratio"] * d_safe:
        a = min(a, p["a_brake_hard"])
    elif d_gap < d_safe:
        a = min(a, p["a_brake_hard"])
    elif rel_speed < p["d_safe_relv_fast"] and d_gap < d_safe * p["d_safe_ratio_fast"]:
        a = min(a, p["a_brake_moderate"])
    elif rel_speed < p["d_safe_relv_moderate"] and d_gap < d_safe * p["d_safe_ratio_moderate"]:
        a = min(a, p["a_brake_light"])

    if ttc < p["ttc_critical"]:
        a = min(a, p["ttc_brake_critical"])
    elif ttc < p["ttc_warning"]:
        a = min(a, p["ttc_brake_warning"])

    if d_gap > max(p["gap_upper_ratio_1"] * d_safe, p["gap_upper_abs_1"]) and rel_speed > p["relv_proactive_1"]:
        a = max(a, p["a_idle"])
    if d_gap > max(p["gap_upper_ratio_2"] * d_safe, p["gap_upper_abs_2"]) and rel_speed > p["relv_proactive_2"]:
        a = max(a, p["a_slight_accel"])
    if d_gap > max(p["gap_upper_ratio_3"] * d_safe, p["gap_upper_abs_3"]) and rel_speed > p["relv_proactive_3"]:
        a = max(a, p["a_moderate_accel"])

    return float(max(a_low, min(a, a_high)))