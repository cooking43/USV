"""Paths and operating constants.

Every value that appears in the paper's hyperparameter appendix is defined
here once, so that the tables and the code cannot drift apart.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

#: Root of the extracted USVTrack release. Set ``USVTRACK_ROOT`` or edit this.
DATASET_ROOT = os.environ.get("USVTRACK_ROOT",
                              os.path.join(os.path.expanduser("~"), "USVTrack"))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Where derived artefacts are written. The benchmark shipped with the
#: repository lives in ``assets/iwdb``; anything rebuilt goes here.
OUTPUT_ROOT = os.environ.get("VLM_USV_OUT",
                             os.path.join(PROJECT_ROOT, "outputs"))
ASSETS_ROOT = os.path.join(PROJECT_ROOT, "assets")

IMAGES_DIR = os.path.join(DATASET_ROOT, "images")      # images/{train,test}/<seq>/...
RADAR_DIR = os.path.join(DATASET_ROOT, "radaruv")      # radaruv/<seq>.csv
YOLO_LABEL_DIR = os.path.join(DATASET_ROOT, "YOLO", "labels")
YOLO_CLASSES = os.path.join(DATASET_ROOT, "YOLO", "classes.txt")


# --------------------------------------------------------------------------
# Sensor geometry
# --------------------------------------------------------------------------

IMG_W, IMG_H = 1920, 1080
RADAR_HZ = 15.0            # measured median sweep interval of the release
RADAR_AZ_LIMIT = 56.5      # deg, half-FOV of the 4D radar
CAM_HFOV_DEG = 110.0       # only used for the fallback bearing estimate


# --------------------------------------------------------------------------
# Perception / association
# --------------------------------------------------------------------------

@dataclass
class PerceptionConfig:
    """Association of radar returns to an image bounding box."""

    bbox_pad_px: float = 8.0        # box dilation before the u,v test
    min_points: int = 3             # returns needed to accept a metric range
    range_gate_m: float = 250.0     # discard returns beyond this
    power_floor_db: float = 0.0     # discard returns below this
    # MPAF similarity scales
    sigma_r: float = 3.0
    sigma_theta: float = 5.0
    sigma_rho: float = 5.0
    sigma_v: float = 3.0
    t_edge: float = 0.55            # T_edge in the paper
    rcs_exponent: float = 4.0       # omega_i = rho_i * r_i**4
    doppler_static_mps: float = 0.35  # |v| below this counts as a static return


# --------------------------------------------------------------------------
# Scene description quantisation (used by both the annotator and the prompt)
# --------------------------------------------------------------------------

# Range bands, metres.  d_m is the outer boundary of the described scene.
D_CLOSE = 30.0
D_MID = 70.0
D_MAX = 120.0            # d_m

# Bearing zones, degrees relative to the bow, positive to starboard.
BEARING_ZONES = (
    ("port_beam", -112.5, -67.5),
    ("port_bow", -67.5, -22.5),
    ("ahead", -22.5, 22.5),
    ("starboard_bow", 22.5, 67.5),
    ("starboard_beam", 67.5, 112.5),
)

APPROACH_EPS = 0.30      # m/s, |range rate| below this counts as steady


# --------------------------------------------------------------------------
# Own-ship model (3-DOF replay simulator)
# --------------------------------------------------------------------------

@dataclass
class VesselConfig:
    """First-order Nomoto steering plus a first-order speed lag."""

    T_yaw: float = 4.1          # s, Nomoto time constant
    K_yaw: float = 0.28         # 1/s, Nomoto gain
    T_speed: float = 3.0        # s, speed lag
    rudder_max_deg: float = 35.0
    yaw_rate_max_dps: float = 8.0
    accel_max: float = 0.6      # m/s^2
    u_nominal: float = 1.5      # m/s, median speed over ground of the release
    length_m: float = 4.5       # own-vessel length
    R0: float = 3.0             # m, contact radius (hull + target half-extent)


# --------------------------------------------------------------------------
# Safety shield
# --------------------------------------------------------------------------

@dataclass
class ShieldConfig:
    d_safe: float = 8.0         # m, required passing margin
    T_h: float = 8.0            # s, prediction horizon
    dt_pred: float = 0.5        # s, horizon discretisation
    v_perp: float = 1.0         # m/s, bound on unmeasured cross-range motion
    varpi: float = 10.0         # weight of the rule-preference term Xi
    alter_grid_deg: tuple = (-45, -35, -25, -15, -10, -5, 0,
                             5, 10, 15, 25, 35, 45)
    speed_grid: tuple = ("hold", "reduce", "half", "stop")


# --------------------------------------------------------------------------
# Knowledge base admission
# --------------------------------------------------------------------------

@dataclass
class MemoryConfig:
    T_red: float = 0.92         # redundancy threshold
    T_sim: float = 0.75         # retrieval threshold; below T_red so that a
                                # case can be retrieved without being redundant
    k_rules: int = 3            # articles retrieved per decision
    k_cases: int = 3            # cases retrieved per decision
    embed_dim: int = 384


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

DECISION_HZ = 2.3           # measured end-to-end rate of the 2B pipeline
SIM_DT = 0.05               # s, integration step
SHIELD_HZ = 20.0            # shield re-evaluation rate


@dataclass
class RunConfig:
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    vessel: VesselConfig = field(default_factory=VesselConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    decision_hz: float = DECISION_HZ
    sim_dt: float = SIM_DT
    seed: int = 0

    def to_dict(self):
        return asdict(self)


DEFAULT = RunConfig()
