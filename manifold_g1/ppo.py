"""Minimal PPO (clipped surrogate + GAE) for the single-environment harness."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _orthogonal_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, np.sqrt(2.0))
        nn.init.zeros_(module.bias)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 128), init_log_std: float = -1.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.Tanh()]
            prev = width
        self.body = nn.Sequential(*layers)
        self.mu = nn.Linear(prev, act_dim)
        self.value = nn.Linear(prev, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        self.apply(_orthogonal_init)
        nn.init.orthogonal_(self.mu.weight, 0.01)
        nn.init.zeros_(self.mu.bias)
        nn.init.orthogonal_(self.value.weight, 1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, obs: torch.Tensor):
        features = self.body(obs)
        return self.mu(features), self.value(features).squeeze(-1)

    def distribution(self, obs: torch.Tensor):
        mu, value = self(obs)
        return torch.distributions.Normal(mu, self.log_std.exp()), value


class RolloutBuffer:
    def __init__(self, steps: int, obs_dim: int, act_dim: int):
        self.obs = np.zeros((steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((steps, act_dim), dtype=np.float32)
        self.logprobs = np.zeros(steps, dtype=np.float32)
        self.rewards = np.zeros(steps, dtype=np.float32)
        self.values = np.zeros(steps, dtype=np.float32)
        self.dones = np.zeros(steps, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, logprob, reward, value, done) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(self.rewards)
        gae = 0.0
        for t in reversed(range(self.ptr)):
            next_value = last_value if t == self.ptr - 1 else self.values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            gae = delta + gamma * lam * next_nonterminal * gae
            advantages[t] = gae
        returns = advantages + self.values[: self.ptr]
        return advantages, returns


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, lr: float = 3e-4, gamma: float = 0.99,
                 lam: float = 0.95, clip: float = 0.2, epochs: int = 10, minibatches: int = 8,
                 value_coef: float = 0.5, entropy_coef: float = 0.005, max_grad_norm: float = 0.5,
                 device: str | None = None):
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.epochs = epochs
        self.minibatches = minibatches
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist, value = self.model.distribution(obs_t)
        action = dist.mean if deterministic else dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        return action.squeeze(0).cpu().numpy(), float(logprob.item()), float(value.item())

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, value = self.model(obs_t)
        return float(value.item())

    def update(self, buffer: RolloutBuffer, last_value: float) -> dict:
        advantages, returns = buffer.compute_gae(last_value, self.gamma, self.lam)
        obs = torch.as_tensor(buffer.obs[: buffer.ptr], device=self.device)
        actions = torch.as_tensor(buffer.actions[: buffer.ptr], device=self.device)
        old_logprobs = torch.as_tensor(buffer.logprobs[: buffer.ptr], device=self.device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        batch = buffer.ptr
        minibatch = max(1, batch // self.minibatches)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}
        for _ in range(self.epochs):
            indices = torch.randperm(batch, device=self.device)
            for start in range(0, batch, minibatch):
                idx = indices[start:start + minibatch]
                dist, value = self.model.distribution(obs[idx])
                logprob = dist.log_prob(actions[idx]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                ratio = (logprob - old_logprobs[idx]).exp()
                unclipped = ratio * adv[idx]
                clipped = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (value - ret[idx]).pow(2).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.item())
                stats["updates"] += 1
        for key in ("policy_loss", "value_loss", "entropy"):
            stats[key] /= max(1, stats["updates"])
        return stats

    def save(self, path) -> None:
        torch.save({"model": self.model.state_dict()}, path)

    def load(self, path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
