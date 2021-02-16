import random
import numpy as np

from common.segment_tree import SumSegmentTree, MinSegmentTree


# Replay Buffer
class ReplayBuffer:
    def __init__(self, size, n_steps):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.n_steps = n_steps

    def add(self, obses_t, actions, rewards, obses_tp1, dones):
        experience = (obses_t, actions, rewards, obses_tp1, dones)
        if self._next_idx >= len(self._storage):
            self._storage.append(experience)
        else:
            self._storage[self._next_idx] = experience
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            experience = self._storage[i]
            obs_t, action, reward, obs_tp1, done = experience
            obses_t.append(obs_t)
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(obs_tp1)
            dones.append(done)
        return np.array(obses_t, copy=False), np.array(actions, copy=False), \
               np.array(rewards, copy=False), np.array(obses_tp1, copy=False), \
               np.array(dones, copy=False)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        # return self._encode_sample(idxes)
        return self._sample_episode(idxes)

    def add_episode(self, obses_t, actions, rewards, dones, fps):
        episode = (obses_t, actions, rewards, dones, fps)
        if self._next_idx >= len(self._storage):
            self._storage.append(episode)
        else:
            self._storage[self._next_idx] = episode
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _sample_episode(self, idxes):
        batch_obses_t, batch_actions, batch_rewards, batch_dones, batch_fps = [], [], [], [], []
        for i in idxes:
            obses_t, actions, rewards, dones, fps = self._sample_steps(self._storage[i])
            batch_obses_t.append(obses_t)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_dones.append(dones)
            batch_fps.append(fps)
        # print(f'batch_dones is {batch_dones}')
        # print(np.array(batch_dones).shape)
        return np.array(batch_obses_t, copy=False), np.array(batch_actions, copy=False), \
               np.array(batch_rewards, copy=False), np.array(batch_dones, copy=False), \
               np.array(batch_fps, copy=False)

    def _sample_steps(self, episode):
        # print(f'{episode[1].shape[1]} - {self.n_steps} = {episode[1].shape[1]- self.n_steps}')
        start = random.randint(0, episode[1].shape[1] - self.n_steps)
        obses_t = episode[0][:, start: start + self.n_steps + 1]  # +1 is to have St+n in n-step return
        actions = episode[1][:, start: start + self.n_steps]
        rewards = episode[2][:, start: start + self.n_steps]
        dones = episode[3][:, start: start + self.n_steps + 1]
        fps = episode[4][:, start: start + self.n_steps]

        return np.array(obses_t, copy=False), np.array(actions, copy=False),\
               np.array(rewards, copy=False), np.array(dones, copy=False), \
               np.array(fps, copy=False)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, n_steps):
        super(PrioritizedReplayBuffer, self).__init__(size, n_steps)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def add_episode(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add_episode(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array([weights], dtype=np.float32)
        # encoded_sample = self._encode_sample(idxes)
        encoded_sample = self._sample_episode(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
