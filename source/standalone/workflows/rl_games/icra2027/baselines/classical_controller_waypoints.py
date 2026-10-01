"""
Classical baseline controller: clause-hauled tacking with a manually-tuned
hybrid thruster assist, in the style of the classical methods described in
the paper's related work (Wille 2016; Erckens et al./Avalon's tack state
machine; Zhang et al.'s manually-tuned tack-assist propeller).

This is a hand-designed control law, not a learned policy -- it computes
actions directly from `info` (the same dict your env's get_info() returns),
bypassing the RL agent entirely. `assist_level` (a fixed thrust magnitude
used during the tack-assist window) plays the same role as the RL policy's
`energy_context`: sweeping it traces out the classical controller's own
energy/time curve for a fair comparison against the learned Pareto front.

Action layout matches your env's convention:
    actions[:, 0] = thruster_left
    actions[:, 1] = thruster_right
    actions[:, 2] = rudder
    actions[:, 3] = sail

--- Refinements from your actual polar diagram and CL/CD curves ---

1. UPWIND CLOSE-HAULED ANGLE is now looked up from a real boat-speed polar
   table (PolarTable below) instead of a fixed guess -- for each wind speed,
   it uses the angle that maximizes VMG = boat_speed * cos(angle), the
   standard real-sailing "optimal tacking angle" definition. Pass the CSV
   that generated your sailing_polar_max_vmg plot (columns: wind_speed,
   wind_angle_deg, boat_speed) via `polar_csv_path`; without one, it falls
   back to a fixed FALLBACK_UPWIND_ANGLE_DEG.

2. DOWNWIND NO-GO ZONE REMOVED BY DEFAULT. Your polar shows a sharp cusp
   (zero VMG) at 0 deg -- confirming a real upwind no-go zone -- but NO cusp
   at 180 deg: the curves stay smoothly rounded there, meaning the boat can
   make continuous progress sailing dead downwind. The previous version of
   this controller wrongly assumed a symmetric downwind dead zone
   (MAX_DOWNWIND_ANGLE_DEG=140) and forced a broad-reach "tack" near 180 deg
   that your own data doesn't support. Set ENABLE_DOWNWIND_NOGO=True below
   if you find real evidence otherwise (e.g. a config-level dead-zone that
   this particular polar sweep didn't exercise).

3. SAIL TRIM now targets the angle of attack that maximizes CL/CD (computed
   directly from your aerodynamics.generate_coeffs(), searched only within
   the XFOIL-fitted range so it can't lock onto a Viterna-only decoy lobe --
   same concern as the earlier stall-boundary bug) instead of the previous
   "half the apparent wind angle" heuristic.

ASSUMPTIONS STILL NEEDING EMPIRICAL VERIFICATION (flagged inline):
  - The mapping from a target angle of attack to the raw [-1,1] sail action
    (MAX_SAIL_ACTION_ANGLE_RAD below) -- verify by commanding a few constant
    sail actions and checking the resulting `sail_angle`/`aoa` in info.
  - RUDDER_KP, STALL_SPEED_THRESHOLD, ASSIST_WIND_ANGLE_DEG are still
    starting guesses, not tuned against your real dynamics.
"""

import torch
import numpy as np
import pandas as pd
import math
# --- Tunable constants -----------------------------------------------------
FALLBACK_UPWIND_ANGLE_DEG = 45.0   # used only if no polar_csv_path is given
MAX_CROSS_TRACK = 10.0             # tack leg lateral budget -- matches cfg.max_cross_track.
                                    # Also used to size each precomputed tack leg (see
                                    # generate_tack_waypoints): leg length is chosen so a
                                    # leg drifts exactly this far sideways before ending.
MAX_TACK_WAYPOINTS = 12            # generous cap on precomputed tack legs -- a typical
                                    # 30-35 unit course only needs a handful at the leg
                                    # lengths this controller picks; see reset()/
                                    # generate_tack_waypoints().
WAYPOINT_SAILING_AWAY_COS_THRESHOLD = 0.1  # stop generating further tack legs once
                                    # continuing in the current leg's direction would
                                    # point more than ~84 deg away from the goal
                                    # (cos(angle) below this) -- the remaining approach
                                    # is snapped straight to the goal instead.
RUDDER_KP = 1.0 #/ np.radians(60.0)     # full rudder command at 60 deg heading error
STALL_SPEED_THRESHOLD = 0.3        # m/s -- below this, boat is considered "stalled"
LOW_SPEED_BANGBANG_THRESHOLD = 0.3 # m/s -- below this, rudder saturates to full
                                    # authority instead of a graded proportional
                                    # response -- see the speed^2 turning-authority
                                    # note in classical_action(). NOTE: this remains
                                    # justified under the REAL rudder model too, for
                                    # a genuinely physical reason now -- hydrodynamic
                                    # force scales with flow speed^2 (F=0.5*rho*V^2*
                                    # A*CL), so turning authority still collapses at
                                    # low boat speed, just via real fluid dynamics
                                    # instead of the old toy model's explicit
                                    # speed^2 term.

# --- Rudder safety scaling, derived from your rudder's own CL curve --------
# (rudder_cl_cd_aoa.png / rudder_cl_cd_ratio.png -- read approximately from
# the plots since no raw CSV was provided for the rudder, unlike the sail's
# exact generate_coeffs()-derived values. Swap in
# compute_rudder_stall_angle_deg() below for an exact number once convenient.)
#
# CL peaks (stall boundary) at roughly +/-15 deg -- past this, force becomes
# unreliable until the much-later Viterna secondary lobe (~105 deg), which is
# not a realistic mechanical rudder deflection. Unlike sail trim (which
# targets max L/D for efficiency, ~8-10 deg here), the rudder's job is
# maximum RELIABLE turning force, so ~15 deg (near max CL) is the right
# target ceiling, not the L/D-optimal angle.
RUDDER_STALL_ANGLE_DEG = 15.0
# Command saturates to this fraction of the actuator's own [-1,1] range,
# NOT to 1.0 -- keeping the worst case (bang-bang at low speed, which always
# asks for full authority) inside the safe pre-stall lift region regardless
# of how wide the actuator's own configured max-angle happens to be.
# ASSUMED_ACTUATOR_MAX_ANGLE_DEG is a placeholder -- replace with the actual
# max deflection from your rudder_actuator_config() (boat_config.py) once
# you have it; until then this assumes a conservative-but-unverified 45 deg
# actuator range, giving a safety scale of 15/45 = 0.33.
# ASSUMED_ACTUATOR_MAX_ANGLE_DEG below should match cfg.scale_joint_pos in
# your rudder_actuator_config(). As of writing, that config had a real bug:
# `cfg.scale_joint_pos = 180.0 * torch.pi / 180.0  # Max +/-35 degrees
# deflection` -- the comment says 35 deg, the value evaluates to 180 deg (a
# 5x mismatch). If you've fixed it to match the comment's stated intent
# (recommended -- 35 deg is also a standard real-world rudder hard-over
# limit), 35.0 below is correct. If you instead deliberately kept the wider
# range, update this to match whatever cfg.scale_joint_pos actually is.
ASSUMED_ACTUATOR_MAX_ANGLE_DEG = 180  # set to match your (corrected) rudder_actuator_config()
RUDDER_SAFETY_SCALE = RUDDER_STALL_ANGLE_DEG / ASSUMED_ACTUATOR_MAX_ANGLE_DEG
ASSIST_WIND_ANGLE_DEG = 20.0       # only assist within this many degrees of the no-go direction
MAX_SAIL_ACTION_ANGLE_RAD = torch.pi   # NEEDS VERIFICATION: assumed raw action
                                            # +/-1 maps linearly to +/- this many
                                            # radians of foil angle -- see caveat above


def wrap_to_pi(angle_rad: torch.Tensor) -> torch.Tensor:
    return (angle_rad + torch.pi) % (2 * torch.pi) - torch.pi


def generate_tack_waypoints(start_pos: torch.Tensor, goal_pos: torch.Tensor, nogo_reference_w: torch.Tensor,
                             tack_side0: torch.Tensor, min_upwind: torch.Tensor, tack_leg_length: torch.Tensor,
                             max_waypoints: int = MAX_TACK_WAYPOINTS) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute a whole zigzag tack-leg sequence from start_pos to goal_pos, decided
    ONCE up front instead of live, step by step.

    This is what actually fixes both the tack-switch chatter and the "sails past the
    goal" overshoot bugs, instead of patching around them: with a reactive approach,
    "when do I flip" and "when do I stop tacking and go straight" are decisions made
    every control step from noisy live state (bearing, cross-track) -- exactly what let
    both bugs happen. Deciding the whole sequence of leg endpoints in advance means
    there's nothing left to decide live: classical_action() just steers at whichever
    waypoint is current and advances an index once it's reached.

    Adapted from the (drafted but never wired up) prototype in
    kingfisher_sail_23_06_25.py's generate_tacking_waypoints(). tack_leg_length is the
    "adaptable" length requested by the caller -- ClassicalControllerState.reset()
    passes MAX_CROSS_TRACK / sin(min_upwind), the along-track distance that drifts
    exactly MAX_CROSS_TRACK sideways at the close-hauled angle, so leg length
    automatically adapts with wind speed (min_upwind is itself wind-speed-dependent,
    from PolarTable) and stays tied to the same lateral budget the env enforces.

    Each leg alternates tack_side (starting from tack_side0) and stops generating
    further legs once ANY of these is true: the goal is no longer upwind at all (its
    direct bearing from the current point already sits outside the no-go cone -- this is
    what stops a leg from ever being generated in the first place when the goal was
    never upwind to begin with, e.g. a 90 deg wind case where the STARTING heading
    happens to be close-hauled-shaped but the goal itself was never in the no-go cone),
    continuing would point mostly away from the goal (WAYPOINT_SAILING_AWAY_COS_THRESHOLD),
    or the next leg would already land within one leg-length of the goal. In all three
    cases the leg's endpoint is snapped exactly to goal_pos, guaranteeing the LAST
    waypoint in the sequence is always the goal itself, never short of or past it.

    Returns:
        waypoints: (N, max_waypoints, 2) -- waypoints[:, 0] is start_pos; entries at or
            past n_valid are unused padding, ignored by the caller.
        n_valid: (N,) int64 -- how many of the leading entries in `waypoints` are real,
            including start_pos at index 0 and the goal as the last valid entry.
    """
    device = start_pos.device
    num_envs = start_pos.shape[0]

    waypoints = torch.zeros((num_envs, max_waypoints, 2), device=device)
    waypoints[:, 0, :] = start_pos
    current_pos = start_pos.clone()
    tack_side = tack_side0.clone()
    finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
    n_valid = torch.ones(num_envs, dtype=torch.int64, device=device)

    for step in range(1, max_waypoints):
        # Is tacking even needed from the CURRENT point? If the goal's direct bearing
        # already sits outside the no-go cone, there's nothing to tack around -- go
        # straight to the goal instead of forcing a leg. Without this check, the loop
        # below would generate a close-hauled leg unconditionally on every call, even
        # when the goal was never upwind in the first place (e.g. a 90 deg wind case
        # where the boat's starting heading happens to look close-hauled-shaped, but the
        # goal itself was never within min_upwind of the no-go direction) -- that
        # produced a real, visible detour for a goal that didn't need tacking at all.
        to_goal_vec = goal_pos - current_pos
        goal_bearing = torch.atan2(to_goal_vec[:, 1], to_goal_vec[:, 0])
        goal_nogo_angle = wrap_to_pi(goal_bearing - nogo_reference_w)
        not_upwind = goal_nogo_angle.abs() >= min_upwind

        tack_angle = nogo_reference_w + tack_side * min_upwind
        tack_dir = torch.stack((torch.cos(tack_angle), torch.sin(tack_angle)), dim=1)
        next_pos = current_pos + tack_leg_length.unsqueeze(-1) * tack_dir

        to_next_vec = next_pos - current_pos
        goal_dir = to_goal_vec / to_goal_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        leg_dir = to_next_vec / to_next_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        goal_proj = (goal_dir * leg_dir).sum(dim=1)

        sailing_away = goal_proj < WAYPOINT_SAILING_AWAY_COS_THRESHOLD
        near_goal = (goal_pos - next_pos).norm(dim=1) < tack_leg_length
        done_now = (~finished) & (not_upwind | sailing_away | near_goal)

        next_pos = torch.where(done_now.unsqueeze(-1), goal_pos, next_pos)
        waypoints[:, step, :] = next_pos
        n_valid = n_valid + (~finished).to(torch.int64)

        finished = finished | done_now
        current_pos = torch.where(finished.unsqueeze(-1), goal_pos, next_pos)
        tack_side = -tack_side

        if finished.all():
            break

    # Safety net: if max_waypoints was exhausted before a natural stop condition
    # fired (shouldn't normally happen given how tack_leg_length/course length
    # scale), force the last generated waypoint to the goal so the following code
    # never ends up treating a non-goal point as the final target.
    if not finished.all():
        last_idx = (n_valid - 1).clamp(max=max_waypoints - 1)
        waypoints[torch.arange(num_envs, device=device), last_idx] = goal_pos

    return waypoints, n_valid


LSA_MIN_ANGLE_DEG = 0.0    # sail sheeted in tight when heading into the wind
LSA_MAX_ANGLE_DEG = 90.0   # sail eased fully out when running downwind


def linear_sail_angle_deg(apparent_wind_angle_deg: torch.Tensor) -> torch.Tensor:
    """
    Linear Sail Angle (LSA) rule -- the standard classical sailboat sail-trim
    heuristic (Jaulin & Le Bars 2013b; Santos et al. 2016; used in VAIMOS,
    Marius, ASR). Sail angle scales LINEARLY with apparent wind angle magnitude:

        delta_s = (delta_s_max - delta_s_min) * (alpha_aw / pi) + delta_s_min

    No CL/CD curve, no optimization, no lookup table -- this REPLACES
    DriveForceOptimalTrim as the active sail-trim law. It's deliberately the
    simplest possible correct rule, which is exactly why it's the standard
    baseline in this literature rather than an "optimal" trim law: fewer
    assumptions, fewer places for a sign/convention bug to hide. It also
    outputs the target FOIL ANGLE directly (delta_s IS the sail angle, not
    an angle-of-attack needing a separate apparent-wind inversion step) --
    one less conversion than the DriveForceOptimalTrim path had, and one
    less place a bug could hide.

    alpha_aw here uses the STANDARD convention (0=no-go/head-to-wind,
    pi=dead downwind). Correction to your env's actual (opposite) raw
    convention reuses PolarTable.NOGO_IS_AT_ZERO as the single source of
    truth, same as everywhere else in classical_action.
    """
    gamma_abs = apparent_wind_angle_deg.abs()
    if not PolarTable.NOGO_IS_AT_ZERO:
        gamma_abs = 180.0 - gamma_abs
    magnitude = (LSA_MAX_ANGLE_DEG - LSA_MIN_ANGLE_DEG) * (gamma_abs / 180.0) + LSA_MIN_ANGLE_DEG
    return -torch.sign(apparent_wind_angle_deg) * magnitude


def compute_rudder_stall_angle_deg(rudder_hydrodynamics, device, search_range_deg=(0.0, 40.0)) -> float:
    """
    EXACT version of the ~15 deg estimate read off rudder_cl_cd_aoa.png --
    call this once you can reach the live rudder hydrodynamics object
    (env.unwrapped._rudder_hydrodynamics, same pattern as _sail_aerodynamics)
    to replace the approximate RUDDER_STALL_ANGLE_DEG constant with a real
    number computed the same way as the sail's stall boundary: the first
    local CL maximum walking outward from 0 deg, so it can't lock onto the
    Viterna secondary lobe further out (same concern as before, just for a
    different foil).
    """
    aoa = torch.linspace(search_range_deg[0], search_range_deg[1], 400, device=device)
    cl, _ = rudder_hydrodynamics.generate_coeffs(aoa)
    n = len(cl)
    for i in range(1, n - 1):
        if cl[i - 1] < cl[i] > cl[i + 1]:
            print(f"[INFO] Rudder stall angle (first local CL max): {aoa[i].item():.1f} deg")
            return aoa[i].item()
    print("[WARNING] No local max found in search range -- falling back to approximate estimate")
    return RUDDER_STALL_ANGLE_DEG


class PolarTable:
    """
    Wind-speed-dependent optimal close-hauled tacking angle, derived
    EMPIRICALLY from a measured VMG polar sweep (e.g. sail_sweep_results.csv,
    the data behind your sailing_polar_max_vmg plot).

    IMPORTANT: earlier versions of this controller assumed the no-go/dead-
    upwind direction sits at angle=0 in whatever convention was being used.
    Checking directly against the real sail_sweep_results.csv data showed
    that assumption was wrong for that CSV's `test_wind_angle` column -- the
    true no-go center (minimum VMG) sits at ~182.5 degrees there, not 0.
    Rather than hardcoding a location, this class DETECTS the no-go center
    directly from the data (the angle of minimum VMG, per wind speed) and
    reports the optimal tacking angle as an OFFSET (angular distance) from
    that detected center. The offset magnitude is what matters physically
    and is robust to whatever angle-labeling convention a given CSV happens
    to use.

    CSV format expected: columns `test_wind_angle` (deg, swept 0-360),
    `test_wind_speed`, `max_vmg` (or override the column names below).

    !! REMAINING RISK, NOT YET VERIFIED: this class tells you the *offset*
    magnitude to steer away from dead-upwind. It does NOT tell you whether
    the env's own `true_wind_angle_w`/`true_wind_angle_b` (used inside
    classical_action, read from get_info() at runtime) treats angle=0 as
    the dead-upwind/no-go direction the same way this CSV's sweep script
    might have. If the CSV was generated with a different wind-angle
    convention than the live env uses (e.g. "direction wind blows toward"
    vs "direction wind blows from" -- an easy 180-degree mixup), applying
    this offset around the env's wind_w=0 would steer the controller
    exactly backwards. VERIFY before trusting: in the live env, force a
    fixed wind condition and confirm the boat's speed actually collapses
    when true_wind_angle_b is near 0 (not near 180) -- e.g. watch `distance`
    stop decreasing / forward speed drop to ~0 while logging true_wind_angle_b.
    If it collapses near 180 instead, set NOGO_IS_AT_ZERO=False below.
    """

    NOGO_IS_AT_ZERO = False  # see verification note above -- UNCONFIRMED for the live env

    def __init__(self, csv_path: str | None, angle_col: str = "test_wind_angle",
                 speed_col: str = "test_wind_speed", vmg_col: str = "max_vmg",
                 fallback_angle_deg: float = FALLBACK_UPWIND_ANGLE_DEG,
                 search_half_width_deg: float = 90.0):
        self.fallback_angle_deg = fallback_angle_deg
        if csv_path is None:
            self.speeds = None
            self.offset_angles = None
            return

        df = pd.read_csv(csv_path)
        speeds, offsets, centers = [], [], []
        for ws, g in df.groupby(speed_col):
            g = g.sort_values(angle_col)
            nogo_center = g.loc[g[vmg_col].idxmin(), angle_col]

            # Search a window around the DETECTED no-go center (not an
            # assumed one) for the angle maximizing VMG projected toward it.
            deviation = ((g[angle_col] - nogo_center + 180) % 360) - 180  # wrap to [-180,180]
            mask = deviation.abs() <= search_half_width_deg
            window = g[mask].copy()
            window["deviation"] = deviation[mask]
            vmg_toward_nogo = window[vmg_col] * np.cos(np.radians(window["deviation"]))
            best_offset = window.loc[vmg_toward_nogo.idxmax(), "deviation"]

            speeds.append(ws)
            offsets.append(abs(best_offset))
            centers.append(nogo_center)

        order = np.argsort(speeds)
        self.speeds = np.array(speeds)[order]
        self.offset_angles = np.array(offsets)[order]
        nogo_centers = np.array(centers)[order]
        print(f"[INFO] PolarTable: detected no-go center per wind speed (CSV convention) = "
              f"{dict(zip(self.speeds, nogo_centers))}")
        print(f"[INFO] PolarTable: optimal tacking OFFSET from no-go per wind speed = "
              f"{dict(zip(self.speeds, self.offset_angles))}")
        print("[WARNING] Verify NOGO_IS_AT_ZERO against the live env before trusting this "
              "-- see PolarTable's docstring.")

    def get_upwind_angle_deg(self, wind_speed: torch.Tensor) -> torch.Tensor:
        if self.speeds is None:
            return torch.full_like(wind_speed, self.fallback_angle_deg)
        ws_np = wind_speed.detach().cpu().numpy()
        angle_np = np.interp(ws_np, self.speeds, self.offset_angles)
        return torch.as_tensor(angle_np, device=wind_speed.device, dtype=wind_speed.dtype)


def compute_max_ld_angle_deg(aerodynamics, device, search_range_deg=(2.0, 30.0)) -> float:
    """
    Angle of attack (deg) that maximizes CL/CD, computed directly from your
    real aerodynamics.generate_coeffs() -- the exact same call used in your
    plotting script. Searched only within `search_range_deg` (the pre-stall/
    XFOIL-fitted region) so it can't lock onto a Viterna-only secondary lobe
    further out -- the same concern as the CL-curve stall-boundary bug fixed
    earlier. Call this ONCE at setup, not every step (it's a static property
    of the wing, not of the current flow state).

    NOTE: max L/D is only the right sail-trim TARGET for close-hauled/beam-
    reach sailing, where the forward drive force is lift-dominated. It is
    NOT correct for reaching/running -- see DriveForceOptimalTrim below,
    which computes the actual apparent-wind-angle-dependent optimum and
    should be preferred over this fixed constant for the sail-trim law.
    This function is kept for reference/comparison and for close-hauled-only
    control laws that don't need the full angle-dependent table.
    """
    aoa = torch.linspace(search_range_deg[0], search_range_deg[1], 200, device=device)
    cl, cd = aerodynamics.generate_coeffs(aoa)
    ld = cl / cd.clamp(min=1e-6)
    idx = torch.argmax(ld)
    best_angle = aoa[idx].item()
    print(f"[INFO] Max L/D angle of attack: {best_angle:.1f} deg (L/D={ld[idx].item():.1f})")
    return best_angle


class DriveForceOptimalTrim:
    """
    Apparent-wind-angle-dependent optimal angle of attack for MAXIMUM
    FORWARD DRIVE FORCE, not just max L/D.

    Why this matters: the forward driving force from the sail is NOT simply
    "more lift is better, less drag is better" at every point of sail. It
    decomposes (standard sailing-aerodynamics identity) as

        F_drive(alpha, gamma) = CL(alpha)*sin(gamma) - CD(alpha)*cos(gamma)

    where gamma is the apparent wind angle from the bow. At gamma close to
    the no-go boundary (beating), sin(gamma) is small and cos(gamma) is near
    1, so minimizing drag matters most -- max L/D is a good approximation
    there. But as gamma grows past ~90 deg (reaching), drive increasingly
    comes from raw lift; and by gamma=180 (dead downwind), F_drive = CD
    alone -- drag IS the thrust, and the correct trim is a fully-stalled,
    high-drag "barn door" angle where L/D is actually negative. A fixed
    max-L/D target is a reasonable approximation through about beam reach
    and actively wrong for deep reaching/running.

    This class precomputes the optimal alpha for a grid of apparent wind
    angles ONCE at setup (from your real aerodynamics.generate_coeffs()),
    then does a cheap per-step interpolation lookup -- avoids re-running the
    coefficient sweep every timestep for every env.

    Search is restricted to alpha in [0, 90] deg (one tack side's physically
    valid trim range) mirrored by sign for the other side -- searching the
    full +/-180 deg range risks locking onto the OPPOSITE tack's lift lobe
    (a real curve has two CL peaks, one per tack), which is not a valid trim
    option while sailing on a given tack. This is the same class of concern
    as the earlier CL-vs-alpha stall-boundary bug, applied here as a range
    restriction rather than a local-extremum search.
    """

    def __init__(self, aerodynamics, device, n_gamma_points: int = 181, n_alpha_points: int = 361):
        gamma_deg = torch.linspace(0.0, 180.0, n_gamma_points, device=device)
        alpha_deg = torch.linspace(0.0, 90.0, n_alpha_points, device=device)
        cl, cd = aerodynamics.generate_coeffs(alpha_deg)
        cd = cd.clamp(min=1e-6)

        gamma_rad = torch.deg2rad(gamma_deg).unsqueeze(1)   # (n_gamma, 1)
        cl_row = cl.unsqueeze(0)                             # (1, n_alpha)
        cd_row = cd.unsqueeze(0)                             # (1, n_alpha)
        f_drive = cl_row * torch.sin(gamma_rad) - cd_row * torch.cos(gamma_rad)  # (n_gamma, n_alpha)

        best_idx = torch.argmax(f_drive, dim=1)              # (n_gamma,)
        self.gamma_table_deg = gamma_deg
        self.alpha_table_deg = alpha_deg[best_idx]

        print("[INFO] DriveForceOptimalTrim table built. Sample points:")
        for g in [0, 20, 45, 90, 135, 180]:
            idx = int(round(g / 180.0 * (n_gamma_points - 1)))
            print(f"    apparent_wind_angle={g:3d} deg -> optimal AoA={self.alpha_table_deg[idx].item():.1f} deg")

    def get_optimal_aoa_deg(self, apparent_wind_angle_deg: torch.Tensor) -> torch.Tensor:
        """
        Vectorized lookup. Input can be signed (-180..180, either side of
        the bow); the sign of the apparent wind angle determines which side
        the sail should be trimmed to, magnitude determines how far via the
        table (built for the 0..180 magnitude range, symmetric by tack).
        Returns a SIGNED angle of attack target matching the input's sign.

        CONVENTION FIX: this table was built assuming the standard TWA
        convention (gamma=0 -> no-go/upwind, gamma=180 -> downwind).
        Confirmed directly against the live env: apparent_wind_angle's own
        raw zero-point is the OPPOSITE (0 -> downwind, 180 -> no-go/upwind) --
        the same flip PolarTable.NOGO_IS_AT_ZERO already exists to handle
        for wind_w/wind_b elsewhere in classical_action, just never applied
        here before. Reusing that single flag (rather than a second,
        separately-tracked constant) keeps both corrections from drifting
        out of sync if the convention is ever re-checked.
        """
        gamma_abs = apparent_wind_angle_deg.abs()
        if not PolarTable.NOGO_IS_AT_ZERO:
            gamma_abs = 180.0 - gamma_abs
        idx = torch.searchsorted(self.gamma_table_deg, gamma_abs.clamp(max=180.0))
        idx = idx.clamp(1, len(self.gamma_table_deg) - 1)
        x0, x1 = self.gamma_table_deg[idx - 1], self.gamma_table_deg[idx]
        y0, y1 = self.alpha_table_deg[idx - 1], self.alpha_table_deg[idx]
        t = (gamma_abs - x0) / (x1 - x0).clamp(min=1e-6)
        magnitude = y0 + t * (y1 - y0)
        return torch.sign(apparent_wind_angle_deg) * magnitude


class ClassicalControllerState:
    """Persistent per-env state the controller needs across steps within an
    episode: the precomputed tack-waypoint sequence for this episode (see
    generate_tack_waypoints()), how many of its entries are real, and which
    one the controller is currently steering toward.

    The whole zigzag (and the final direct-to-goal leg) is decided ONCE, at
    reset, instead of reactively re-evaluated every control step from live
    bearing/cross-track state -- that reactive approach was what caused both
    the tack-switch chatter (confirmed via a live debug log: desired_heading
    alternating between its two close-hauled values every single step) and
    the "sails straight through the goal before ever tacking again" overshoot.
    Both were downstream symptoms of deciding leg endpoints live; committing
    to them in advance removes both failure modes at the root."""

    def __init__(self, num_envs: int, device):
        self.waypoints = torch.zeros(num_envs, MAX_TACK_WAYPOINTS, 2, device=device)
        self.n_valid = torch.ones(num_envs, dtype=torch.int64, device=device)
        self.waypoint_idx = torch.ones(num_envs, dtype=torch.int64, device=device)

    def reset(self, env_ids: torch.Tensor, info: dict, polar_table: "PolarTable"):
        """Call whenever the given env_ids begin a new episode. (Re)generates
        each reset env's whole tack-waypoint sequence for the episode ahead."""
        start_pos = info["robot_pos_w"][env_ids, :2].clone()
        goal_pos = info["goal_pos"][env_ids]

        wind_w = torch.deg2rad(info["true_wind_angle_w"][env_ids])
        nogo_shift = 0.0 if PolarTable.NOGO_IS_AT_ZERO else torch.pi
        nogo_reference_w = wrap_to_pi(wind_w + nogo_shift)
        min_upwind = torch.deg2rad(polar_table.get_upwind_angle_deg(info["true_wind_speed"][env_ids]))
        # Along-track distance that drifts exactly MAX_CROSS_TRACK sideways at the
        # close-hauled angle -- see generate_tack_waypoints()'s docstring.
        tack_leg_length = MAX_CROSS_TRACK / torch.sin(min_upwind).clamp(min=1e-3)
        tack_side0 = torch.ones_like(min_upwind)

        waypoints, n_valid = generate_tack_waypoints(
            start_pos, goal_pos, nogo_reference_w, tack_side0, min_upwind, tack_leg_length,
        )
        self.waypoints[env_ids] = waypoints
        self.n_valid[env_ids] = n_valid
        self.waypoint_idx[env_ids] = 1  # first real target -- index 0 is start_pos itself


def classical_action(info: dict, state: ClassicalControllerState, assist_level) -> torch.Tensor:
    """Compute a (num_envs, 4) action tensor for one step. Sail trim uses the
    Linear Sail Angle (LSA) rule -- drive_trim/DriveForceOptimalTrim is no
    longer needed as an argument here. Tacking follows the waypoint sequence
    ClassicalControllerState.reset() precomputed for this episode -- no
    polar_table needed here anymore, only at reset (where the waypoints'
    close-hauled legs are decided once)."""
    device = info["robot_pos_w"].device

    heading_w = torch.deg2rad(info["heading_w"])
    wind_w = torch.deg2rad(info["true_wind_angle_w"])
    wind_b = torch.deg2rad(info["true_wind_angle_b"])
    app_wind_b = torch.deg2rad(info["app_wind_angle"])
    robot_pos = info["robot_pos_w"][:, :2]
    forward_speed = info["lin_vel_b"][:, 0]
    initial_distance = info["initial_distance"]
    distance = info["distance"]

    assist_angle = torch.deg2rad(torch.tensor(ASSIST_WIND_ANGLE_DEG, device=device))

    # NOGO_IS_AT_ZERO is UNVERIFIED against the live env -- see PolarTable's
    # docstring. If wrong, everything below points exactly backwards, so
    # this flip is applied consistently everywhere an angle is compared
    # against the no-go direction (only the thruster-assist trigger uses it
    # directly here now -- the tacking geometry itself was already resolved
    # once, at reset, inside generate_tack_waypoints()).
    nogo_shift = 0.0 if PolarTable.NOGO_IS_AT_ZERO else torch.pi

    # 1. Steer toward the current precomputed tack waypoint (see
    #    ClassicalControllerState's docstring for why this replaced the old
    #    reactive bearing/cross-track logic). Check progress along the
    #    CURRENT leg and advance the target index first, before computing
    #    this step's heading -- otherwise the boat would spend one extra
    #    control step still aiming at an already-reached waypoint.
    env_idx = torch.arange(robot_pos.shape[0], device=device)
    current_target = state.waypoints[env_idx, state.waypoint_idx]
    prev_waypoint = state.waypoints[env_idx, state.waypoint_idx - 1]

    # Along-track progress on this leg, same reasoning as the goal-overshoot
    # fix this replaced: robust to the boat not passing exactly through the
    # waypoint, not just to reaching it exactly.
    leg_vec = current_target - prev_waypoint
    leg_len = torch.norm(leg_vec, dim=1).clamp(min=1e-6)
    leg_unit = leg_vec / leg_len.unsqueeze(-1)
    progress = torch.sum((robot_pos - prev_waypoint) * leg_unit, dim=1)
    reached_target = progress >= leg_len

    can_advance = state.waypoint_idx < (state.n_valid - 1)
    state.waypoint_idx = torch.where(reached_target & can_advance, state.waypoint_idx + 1, state.waypoint_idx)

    # The last waypoint in every env's sequence is always the goal itself
    # (guaranteed by generate_tack_waypoints), so once waypoint_idx reaches
    # it, "steer at the current waypoint" already IS "steer at the goal" --
    # no separate direct-to-goal branch needed.
    current_target = state.waypoints[env_idx, state.waypoint_idx]
    target_vec = current_target - robot_pos
    desired_heading = torch.atan2(target_vec[:, 1], target_vec[:, 0])

    # 4. Rudder: proportional heading controller, saturating to FULL authority
    #    whenever speed is low. This matters specifically because of how the
    #    rudder is currently modeled in the env: it's a direct angular-
    #    velocity override scaled by speed^2 (`new_velocities[:,5] = 4*action
    #    *speed**2`), not a torque through inertia. That means turning
    #    authority collapses quadratically as the boat slows down during a
    #    tack -- exactly the situation a proportional law handles worst,
    #    since it would otherwise command LESS than max rudder at the exact
    #    moment every bit of remaining authority is needed to avoid getting
    #    stuck head-to-wind ("in irons"). Below LOW_SPEED_BANGBANG_THRESHOLD,
    #    just slam to +/-1 in the direction of the heading error instead of
    #    a graded response.
    # 4. Rudder: proportional heading controller, saturating to a SAFE
    #    fraction of full authority whenever speed is low.
    #
    #    Two things matter here now that the real hydrodynamic rudder is
    #    active (not the old speed^2 velocity-override toy model):
    #
    #    (a) Turning authority still collapses at low boat speed, but now
    #        for a genuine physical reason -- hydrodynamic force scales with
    #        flow speed^2 (F = 0.5*rho*V^2*A*CL), so the low-speed bang-bang
    #        saturation from before remains justified, just on real physics
    #        instead of the toy model's explicit speed^2 term.
    #
    #    (b) Bang-bang saturation always asks for +/-1, the actuator's raw
    #        command range. If the actuator's own configured max deflection
    #        is wider than the rudder's real stall boundary (~15 deg per
    #        rudder_cl_cd_aoa.png), a saturated command could land in the
    #        unreliable post-stall region right when maximum dependable
    #        force is needed most. RUDDER_SAFETY_SCALE keeps the WORST CASE
    #        (full bang-bang command) inside the safe pre-stall lift range --
    #        see the constant's definition for the actuator-range assumption
    #        this depends on, which still needs verifying against your real
    #        rudder_actuator_config().
    heading_error = wrap_to_pi(desired_heading - heading_w)/torch.pi
    proportional_rudder = torch.clamp(RUDDER_KP * heading_error, -1.0, 1.0)
    bangbang_rudder = torch.sign(heading_error)
    low_speed = forward_speed < LOW_SPEED_BANGBANG_THRESHOLD
    rudder_action = proportional_rudder #torch.where(low_speed, bangbang_rudder, proportional_rudder) #* RUDDER_SAFETY_SCALE
    print(f"[DEBUG] desired_heading: {desired_heading.rad2deg()} rudder_action: {rudder_action} speed: {forward_speed}")
    # 5. Sail trim: Linear Sail Angle (LSA) rule -- sheeted in tight
    #    (0 deg) heading into the wind, eased fully out (90 deg) running
    #    downwind. delta_s IS the target foil angle directly -- no
    #    apparent-wind inversion step needed (unlike the old
    #    DriveForceOptimalTrim path, which computed a target angle of
    #    attack and then had to invert through app_wind_b to get a foil
    #    angle -- one fewer place for a convention bug to hide).
    app_wind_b_deg = info["app_wind_angle"]
    target_sail_angle_deg = linear_sail_angle_deg(app_wind_b_deg)
    target_sail_angle_rad = torch.deg2rad(target_sail_angle_deg)
    sail_action = torch.clamp(target_sail_angle_rad / MAX_SAIL_ACTION_ANGLE_RAD, -1.0, 1.0)

    # 6. Hybrid thruster assist: fire only when stalled near the eye of the
    #    wind. wind_b is body-frame, so apply the same nogo_shift correction
    #    used above for the world-frame heading logic.
    wind_b_nogo_relative = wrap_to_pi(wind_b + nogo_shift)
    in_stall_zone = wind_b_nogo_relative.abs() < assist_angle
    is_stalled = forward_speed < STALL_SPEED_THRESHOLD
    assist_trigger = (in_stall_zone & is_stalled).float()
    #thrust_cmd = assist_trigger * assist_level
    max_thrust_cmd = torch.sqrt(assist_level)*torch.ones_like(assist_trigger)
    thrust_cmd = distance/initial_distance

    #print(f"max_thruster: {max_thrust_cmd} thrust_cmd: {thrust_cmd}")
    thruster_left = torch.clamp(thrust_cmd, max=1)
    thruster_right = torch.clamp(thrust_cmd, max=1)

    actions = torch.stack([thruster_left, thruster_right, rudder_action, sail_action], dim=1)
    return actions.clamp(-1.0, 1.0)