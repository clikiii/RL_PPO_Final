import gym
import numpy as np
import cv2


def custom_reward(state_img):
    """
    state_img is a gray scale img whose size is [84, 84]
    """
    img_left = state_img[61:77, 34:39] # check 5 pixel on the left/right the car
    img_right = state_img[61:77, 45:50]
    std_avg = 0.5*(np.std(img_left) + np.std(img_right))

    if np.mean(img_left) > 115 and np.mean(img_right) > 115: return -0.02
    if std_avg > 5: return -0.05
    return 0

class EnvWrapper(gym.Wrapper):
    def __init__(
        self, 
        env, 
        track_num = 1000, 
        frame_num = 4,
        seed = None
    ):
        super().__init__(env)
        self.track_num = track_num
        self.frame_num = frame_num
        self.observation_space = gym.spaces.Box(
            low=0.0, high=255.0, shape=(frame_num,) + (84, 84), dtype=np.float32
        )
        self.random_generator = np.random.default_rng(seed=seed)

    def process_state_img(self, state):
        state = state[:84, 6:-6]
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

        return state

    def reset(self, seed=None):
        if seed == None:
            seed = self.random_generator.integers(self.track_num).item()

        state, info = self.env.reset(seed=seed)
        state = self.process_state_img(state)
        return np.expand_dims(state, axis=0).repeat(self.frame_num, axis=0), info

    def step(self, action):
        next_state = np.empty(self.observation_space.shape)
        total_reward = 0
        terminated = False
        truncated = False
        # for i in range(1, self.frame_num):
        # for i in range(0, self.frame_num, 2):
        for i in range(self.frame_num):
            if not terminated and not truncated:
                next_state_i, reward_i, terminated, truncated, _ = self.env.step(action)
                next_state_i = self.process_state_img(next_state_i)
                total_reward += reward_i
                    # + custom_reward(next_state_i)
                    # + np.clip(action[1], 0, 1)*0.01
                    # - np.clip(action[2], 0, 1)*0.01)
            next_state[i] = next_state_i
            # next_state[i+1] = next_state_i

        return next_state, total_reward, terminated, truncated
