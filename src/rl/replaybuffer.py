import numpy as np

from stable_baselines3.common.buffers import ReplayBuffer


class ReplayBufferIgnoreTrunc(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Separate storage for terminated and truncated
        self.buf_terminated = np.zeros((self.buffer_size,), dtype=np.bool_)
        self.buf_truncated  = np.zeros((self.buffer_size,), dtype=np.bool_)

    def add(self, obs, next_obs, action, reward, done, infos):
        """
        infos: list of dicts (one per env). We assume n_envs=1.
        """
        # Extract true termination from done, or use the Gym returned terminated if available
        info = infos[0] if isinstance(infos, (list, tuple)) else infos

        # If you want to log truncations, read from info["TimeLimit.truncated"]
        truncated = info.get("TimeLimit.truncated", False)

        # Store flags
        self.buf_truncated[self.pos] = truncated

        # done_for_buffer = only real terminations, ignore truncations
        done_for_buffer = done and not truncated
        self.buf_terminated[self.pos] = done_for_buffer
        super().add(obs, next_obs, action, reward, done_for_buffer, infos)