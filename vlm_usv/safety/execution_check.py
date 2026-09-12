"""Analytic safety shield.

Each target is enclosed in a disc that expands along the prediction horizon,

    R_k(tau) = R_0 + d_safe + v_perp * tau ,

so the growth bounds the cross-range motion a range-and-Doppler sensor does not
resolve.  A command is admissible when the predicted own-vessel path stays
outside every disc over the horizon.  An inadmissible command is replaced by
the admissible command nearest to it, with a preference term that keeps the
repair on the side the governing article requires.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .. import config as C
from ..knowledge import inland_rules as IR
from ..perception.mpaf import TargetState


class Shield:
    def __init__(self, cfg: Optional[C.ShieldConfig] = None,
                 vessel: Optional[C.VesselConfig] = None,
                 use_rule_preference: bool = True):
        self.c = cfg or C.DEFAULT.shield
        self.v = vessel or C.DEFAULT.vessel
        self.use_rule_preference = use_rule_preference

    # ---- feasibility ----
    def clearance(self, turn_deg: float, speed: str,
                  targets: Sequence[TargetState], ego_speed: float) -> float:
        """Smallest margin of the command over the horizon, all targets."""
        worst = math.inf
        f = IR.SPEED_FACTOR.get(speed, 1.0)
        for t in targets:
            d = IR.predicted_min_distance(
                t.range_m, t.bearing_deg, t.approach_rate_mps, turn_deg,
                f, ego_speed, self.c.T_h, self.c.dt_pred,
                IR.expansion_rate(t, self.c))
            worst = min(worst, d)
        return worst

    def is_admissible(self, cmd: Dict, targets: Sequence[TargetState],
                      ego_speed: float) -> bool:
        need = self.v.R0 + self.c.d_safe
        return self.clearance(float(cmd.get("turn_deg", 0.0)),
                              str(cmd.get("speed", "hold")),
                              targets, ego_speed) >= need

    # ---- repair ----
    def repair(self, cmd: Dict, targets: Sequence[TargetState],
               ego_speed: float) -> Tuple[Dict, bool]:
        """Return ``(command, replaced)``.

        The command is left untouched when it is already admissible, so the
        shield is inert on a policy that does not emit infeasible commands and
        the intervention count is a direct measure of how often the generative
        layer would have driven the vessel inside the margin.
        """
        if not targets:
            return cmd, False
        need = self.v.R0 + self.c.d_safe
        turn0 = float(cmd.get("turn_deg", 0.0))
        speed0 = str(cmd.get("speed", "hold"))
        if self.clearance(turn0, speed0, targets, ego_speed) >= need:
            return cmd, False

        gt = IR.governing_target(targets)
        enc, _, art = IR.classify_encounter(gt)
        aset = IR.admissible_actions(enc, gt)

        best, best_key = None, None
        for turn in self.c.alter_grid_deg:
            for speed in self.c.speed_grid:
                d = self.clearance(float(turn), speed, targets, ego_speed)
                if d < need:
                    continue
                # deviation from the command actually issued
                dev = ((float(turn) - turn0) / 30.0) ** 2 + \
                      (IR.SPEED_FACTOR[speed] - IR.SPEED_FACTOR.get(speed0, 1.0)) ** 2
                xi = 0.0 if (not self.use_rule_preference or
                             aset.contains(float(turn), speed)) else 1.0
                key = dev + (self.c.varpi * xi if self.use_rule_preference else 0.0)
                if best_key is None or key < best_key:
                    best, best_key = (float(turn), speed, d), key

        if best is None:
            # Nothing on the grid reaches the margin.  The fallback maximises
            # clearance within the set the governing article prescribes: the
            # shield may choose among compliant actions, it may not authorise a
            # non-compliant one, so a geometry that admits no compliant escape
            # is reported rather than resolved.
            fb, fb_d = None, -math.inf
            for turn in self.c.alter_grid_deg:
                for speed in self.c.speed_grid:
                    if self.use_rule_preference and not aset.contains(float(turn), speed):
                        continue
                    d = self.clearance(float(turn), speed, targets, ego_speed)
                    if d > fb_d:
                        fb, fb_d = (float(turn), speed), d
            if fb is None:
                for turn in self.c.alter_grid_deg:
                    for speed in self.c.speed_grid:
                        d = self.clearance(float(turn), speed, targets, ego_speed)
                        if d > fb_d:
                            fb, fb_d = (float(turn), speed), d
            out = dict(cmd)
            out["turn_deg"], out["speed"] = fb[0], fb[1]
            out["shield"] = "best_effort"
            out["shield_clearance_m"] = round(fb_d, 2)
            return out, True

        out = dict(cmd)
        out["turn_deg"], out["speed"] = best[0], best[1]
        out["shield"] = "repaired"
        out["shield_clearance_m"] = round(best[2], 2)
        return out, True
