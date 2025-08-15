from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import pandas as pd

@dataclass
class IndentPolicy:
    service_level: float = 0.9
    moq: int = 1
    multiple: int = 1
    shelf_life_days: Optional[int] = None  # cap coverage if perishable
    reorder_point_days: int = 7            # fallback if no quantile

_Z_FOR = {0.8:0.8416, 0.85:1.036, 0.9:1.2816, 0.95:1.6449, 0.975:1.96, 0.99:2.326}

def recommend_order(
    daily_mean_demand: float,
    daily_std_demand: float,
    lead_time_days: int,
    on_hand: float,
    policy: IndentPolicy
) -> Dict:
    z = _Z_FOR.get(round(policy.service_level,3), 1.28)
    lead_mean = daily_mean_demand * lead_time_days
    lead_std = daily_std_demand * np.sqrt(lead_time_days)
    safety = z * lead_std
    reorder_point = lead_mean + safety
    qty = max(0.0, reorder_point - on_hand)
    # Round to multiple & MOQ
    if policy.multiple > 1:
        qty = policy.multiple * int(np.ceil(qty / policy.multiple))
    if policy.moq and qty < policy.moq and qty > 0:
        qty = policy.moq
    return dict(
        reorder_point=reorder_point,
        safety_stock=safety,
        suggested_order=float(qty),
        bound_reason="multiple/moq_applied" if (policy.multiple>1 or (policy.moq and qty>0 and qty<policy.moq)) else "reorder_gap"
    )
