import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from network import ActorCritic


class PPOAgent:
    def __init__(
        self,
        model: ActorCritic,
        lr = 5e-5,
        epoch_num = 10,
        batch_size = 250,
        clip_epsilon = 0.2,
        critic_weight = 0.6,
        ent_weight = 0.01,
    ):
        self.model = model
        self.lr = lr
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.clip_epsilon = clip_epsilon
        self.critic_weight = critic_weight
        self.entropy_weight = ent_weight

        # self.actor_optimizer = torch.optim.Adam(model.actor_head.parameters(), lr=lr, eps=1e-5)
        # self.critic_optimizer = torch.optim.Adam(model.critic_head.parameters(), lr=lr, eps=1e-5)
        self.optimizer = torch.optim.Adam(model.actor_head.parameters(), lr=lr, eps=1e-5)

    def learn(
        self,
        state_arr,
        action_arr,
        action_probs_arr,
        return_arr,
        adv_arr,
        cur_epi
    ):
        for _ in range(self.epoch_num):
            for sample_idx in BatchSampler(SubsetRandomSampler(range(state_arr.size()[0])), self.batch_size, False):

                one_state = state_arr[sample_idx]
                one_action = action_arr[sample_idx]
                one_action_prob = action_probs_arr[sample_idx]
                one_return = return_arr[sample_idx]
                one_advantage = adv_arr[sample_idx]

                new_action_probs, entropies, new_state_values = self.model.evaluate(one_state, one_action)
                entropy = entropies.mean()

                ratio = (new_action_probs - one_action_prob).exp()
                actor_loss = -torch.min(
                    one_advantage * ratio,
                    one_advantage * ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon),
                ).mean()
                actor_loss = actor_loss - self.entropy_weight * entropy

                critic_loss = (F.mse_loss(one_return, new_state_values)).mean()
                critic_loss = critic_loss

                # self.actor_optimizer.zero_grad()
                # self.critic_optimizer.zero_grad()
                # actor_loss.backward()
                # critic_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.actor_head.parameters(), 0.5)
                # # torch.nn.utils.clip_grad_norm_(self.model.critic_head.parameters(), 0.5)
                # self.actor_optimizer.step()
                # self.critic_optimizer.step()

                loss = (
                    actor_loss
                    + self.critic_weight * critic_loss
                    - self.entropy_weight * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        self.lr_decay(cur_epi)

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()
    
    def lr_decay(self, cur_epi, total_epi=400):
        new_lr = self.lr * (1 - cur_epi / total_epi)
        # for p in self.actor_optimizer.param_groups: p["lr"] = new_lr
        # for p in self.critic_optimizer.param_groups: p["lr"] = new_lr
        for p in self.optimizer.param_groups: p["lr"] = new_lr

