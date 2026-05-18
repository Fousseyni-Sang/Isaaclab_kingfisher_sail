import torch
import math

class ParkingCurriculum:
    """
    A curriculum scheduler for gradually increasing:
      - goal distance
      - goal orientation range

    The curriculum is episode-based: as num_episodes increases,
    the sampled goals become harder.
    """

    def __init__(
        self,
        device,
        min_dist_start=0.5,
        max_dist_final=40.0,
        min_angle_start=5.0 * math.pi/180,
        max_angle_final=math.pi,
        curriculum_duration=200_000,   # number of episodes to reach full difficulty
    ):
        self.device = device

        # Distance curriculum
        self.min_dist_start = min_dist_start
        self.max_dist_final = max_dist_final

        # Orientation curriculum
        self.min_angle_start = min_angle_start
        self.max_angle_final = max_angle_final

        # How long the curriculum lasts
        self.curriculum_duration = curriculum_duration

    def _progress(self, num_episodes):
        """
        Returns a scalar in [0, 1] representing curriculum progress.
        """
        return min(1.0, num_episodes / self.curriculum_duration)

    def goal_sampler(self, env_ids, num_episodes):
        """
        Input:
            env_ids: tensor of environment indices
            num_episodes: tensor of episode counters for each env

        Output:
            target_distance: tensor [len(env_ids)]
            target_bearing: tensor [len(env_ids)]
            target_orientation: tensor [len(env_ids)]
        """

        # Compute curriculum progress per environment
        p = self._progress(num_episodes.float()).to(self.device)

        # --- Distance curriculum ---
        # Distance grows from min_dist_start → max_dist_final
        max_dist = self.min_dist_start + p * (self.max_dist_final - self.min_dist_start)

        # Sample distance uniformly in [0, max_dist]
        target_distance = torch.rand_like(p) * max_dist

        # --- Bearing (goal direction) ---
        # Always full range [-pi, pi]
        target_bearing = torch.rand_like(p) * 2 * math.pi - math.pi

        # --- Orientation curriculum ---
        # Orientation noise grows from small → full [-pi, pi]
        max_angle = self.min_angle_start + p * (self.max_angle_final - self.min_angle_start)

        # Sample orientation uniformly in [-max_angle, max_angle]
        target_orientation = (torch.rand_like(p) * 2 - 1) * max_angle

        return target_distance, target_bearing, target_orientation
