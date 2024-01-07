import gym
import numpy as np
import torch

from env_wrapper import EnvWrapper
from network import ActorCritic


def test(filepathname):
    rng = np.random.default_rng(seed=315)
    track_seeds = rng.choice(2**32 - 1, size=50, replace=False)
    env = EnvWrapper(
        gym.make("CarRacing-v2", domain_randomize=False, render_mode="human")
    )
    model = ActorCritic(env.observation_space.shape, env.action_space.shape[0], "cpu")
    checkpoint = torch.load(filepathname)
    model.load_state_dict(checkpoint["model"])

    score = 0
    state, _ = env.reset(seed=track_seeds[0].item())
    while True:
        env.render()

        state_set = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = model.forward(state_set)
        next_state, reward, terminated, truncated = env.step(action)

        score += reward
        state = next_state
        if terminated:
            break
    print("Score:", score)

if __name__ == "__main__":
    test("./best_models/checkpoint_300_2_672.pt")
