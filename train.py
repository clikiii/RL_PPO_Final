import os

import gym
import numpy as np
import torch
import imageio

from ppo import PPOAgent
from env_wrapper import EnvWrapper
from network import ActorCritic


def get_gif(model, filename, device):
    env = EnvWrapper(gym.make("CarRacing-v2", domain_randomize=False, render_mode="rgb_array"))

    # model.actor_head.to(device)
    # model.critic_head.to(device)
    model.to(device)
    frames = []
    state, _ = env.reset()
    while True:
        frames.append(env.render())

        with torch.no_grad():
            state_set = torch.tensor(
                state, 
                dtype=torch.float32, 
            ).unsqueeze(0).to(device)
            action, _, _ = model.forward(state_set)

        next_state, _, terminated, truncated = env.step(action)

        # next_state[0] = state[-1]
        state = next_state
        if terminated or truncated:
            break

    if not os.path.exists("./model/gif_9"):
        os.makedirs("./model/gif_9")
    imageio.mimwrite(f"./model/gif_9/{filename}", frames)


def calculate_discounted_returns(reward_arr, discount_factor):
    discounted_returns = np.empty_like(reward_arr)
    discounted_returns[-1] = reward_arr[-1]
    for i in reversed(range(reward_arr.shape[0] - 1)):
        discounted_returns[i] = (
            discounted_returns[i + 1] * discount_factor + reward_arr[i]
        )

    return discounted_returns


def train(
        env, 
        model: ActorCritic, 
        agent: PPOAgent, 
        device, 
        epi_num = 400, 
        discount_factor = 0.99,
        gamma = 0.9
    ):
    if not os.path.exists("./model_9"):
        os.makedirs("./model_9")

    max_steps = (1000 // env.frame_num)

    for episode in range(epi_num):
        state_arr = np.zeros((max_steps+1, *env.observation_space.shape), dtype=np.float32)
        action_arr = np.zeros((max_steps+1, env.action_space.shape[0]), dtype=np.float32)
        action_prob_arr = np.zeros((max_steps+1,), dtype=np.float32)
        state_val_arr = np.zeros((max_steps+1,), dtype=np.float32)
        reward_arr = np.zeros((max_steps+1,), dtype=np.float32)

        step_cnt = max_steps

        state, _ = env.reset()
        for step in range(max_steps):
            state_set = torch.tensor(
                state,
                dtype=torch.float32,
            ).unsqueeze(0).to(device)
            action, action_prob, state_value = model.forward(state_set)

            next_state, reward, terminated, truncated = env.step(action)

            state_arr[step] = state
            action_arr[step] = action
            action_prob_arr[step] = action_prob
            state_val_arr[step] = state_value
            reward_arr[step] = reward

            # next_state[0] = state[-1]
            state = next_state

            if terminated or truncated:
                step_cnt = step + 1

                state_set = torch.tensor(
                    state,
                    dtype=torch.float32,
                ).unsqueeze(0).to(device)
                action, action_prob, state_value = model.forward(state_set)

                state_arr[step_cnt] = state
                action_arr[step_cnt] = action
                action_prob_arr[step_cnt] = action_prob
                state_val_arr[step_cnt] = state_value
                reward_arr[step_cnt] = 0
                break

        action_arr = torch.tensor(action_arr[:step_cnt]).to(device)
        action_prob_arr = torch.tensor(action_prob_arr[:step_cnt]).to(device)
        state_arr = torch.tensor(state_arr[:step_cnt]).to(device)
        next_state_val_arr = torch.tensor(state_val_arr[1:step_cnt+1]).to(device)
        state_val_arr = torch.tensor(state_val_arr[:step_cnt]).to(device)
        
        reward_arr = calculate_discounted_returns(reward_arr[:step_cnt], discount_factor)
        reward_arr = (reward_arr - reward_arr.mean()) / (reward_arr.std() + 1e-6)
        reward_arr = torch.tensor(reward_arr).to(device)

        adv_arr = reward_arr + 0.9*next_state_val_arr - state_val_arr
        # adv_arr = reward_arr - state_val_arr
        adv_arr = ((adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-5))

        loss, actor_loss, critic_loss, entropy = agent.learn(
            state_arr, 
            action_arr, 
            action_prob_arr, 
            reward_arr + gamma*next_state_val_arr, 
            adv_arr, 
            episode
        )

        total_reward = reward_arr[:step_cnt].sum()
        print(
            f"[Episode {episode + 1}/{epi_num}] Loss = {loss}, ",
            f"Actor Loss = {actor_loss}, Critic Loss = {critic_loss} ",
            f"Entropy = {entropy}",
            f"Total Reward = {total_reward}",
        )

        if (episode + 1) % 50 != 0:
            print("Saving...")
            torch.save(
                {
                    "it": episode + 1,
                    # "actor_model": model.actor_head.cpu().state_dict(),
                    # "critic_model": model.critic_head.cpu().state_dict()
                    "model": model.cpu().state_dict(),
                },
                f"./model_9/checkpoint_{episode + 1}.pt",
            )
            get_gif(model, f"train_{episode + 1}.gif", device)
            print("Done!")


def main():
    SEED = 100
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"

    env = EnvWrapper(gym.make("CarRacing-v2", domain_randomize=False), seed=SEED)
    model = ActorCritic(env.observation_space.shape, env.action_space.shape[0], device=device)
    # checkpoint = torch.load(f"./model/model_10/checkpoint_0350.pt")
    # model.load_state_dict(checkpoint["model"])
    agent = PPOAgent(model)

    train(env, model, agent, device)
    env.close()


if __name__ == "__main__":
    main()
