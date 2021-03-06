
import random

import numpy as np

from common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size, n_steps):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.n_steps = n_steps

    def __len__(self):
        return len(self._storage)

    def add_episode(self, episode):
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
            # batch_obses_tp1.append(obses_tp1)
            batch_dones.append(dones)
            batch_fps.append(fps)

        return np.array(batch_obses_t, copy=False), np.array(batch_actions, copy=False), \
               np.array(batch_rewards, copy=False), np.array(batch_dones, copy=False),\
               np.array(batch_fps, copy=False)

    def _sample_steps(self, episode):
        # print(f'{episode[1].shape[1]} - {self.n_steps} = {episode[1].shape[1]- self.n_steps}')
        start = random.randint(0, episode[1].shape[1] - self.n_steps)
        obses_t = episode[0][:, start: start + self.n_steps]
        actions = episode[1][:, start: start + self.n_steps]
        rewards = episode[2][:, start: start + self.n_steps]
        # obses_tp1.append(episode[3, :, start: start + self.n_steps, :])
        dones = episode[3][:, start: start + self.n_steps]
        fps = episode[4][:, start: start + self.n_steps]

        return np.array(obses_t, copy=False), np.array(actions, copy=False),\
               np.array(rewards, copy=False), np.array(dones, copy=False), \
               np.array(fps, copy=False)

    def add(self, experience):
        if self._next_idx >= len(self._storage):
            self._storage.append(experience)
        else:
            self._storage[self._next_idx] = experience
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, fps = [], [], [], [], [], []
        for i in idxes:
            experience = self._storage[i]
            obs_t, action, reward, obs_tp1, done, fp = experience
            obses_t.append(obs_t)
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(obs_tp1)
            dones.append(done)
            fps.append(fp)
        return np.array(obses_t, copy=False), np.array(actions, copy=False),\
               np.array(rewards, copy=False), np.array(obses_tp1, copy=False),\
               np.array(dones, copy=False), np.array(fps, copy=False)



    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        # return self._encode_sample(idxes)
        return self._sample_episode(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
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
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
