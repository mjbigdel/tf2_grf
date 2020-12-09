import tensorflow as tf
from ray.new_dashboard import agent
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = keras.Sequential([
        # keras.layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4), activation=tf.nn.relu),
        keras.layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2), activation=tf.nn.relu),
        keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.relu),
        keras.layers.Flatten(),
        keras.layers.Dense(fc1_dims, activation='relu'),
        keras.layers.Dense(fc2_dims, activation='relu'),
        keras.layers.Dense(n_actions, activation=None)])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    return model


class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=1e-2, epsilon_end=0.01,
                 mem_size=100, fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)

            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # q_max = np.max(q_next, axis=1) * dones
        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * (1.0 - dones)

        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def evaluate(self):
        env = football_env.create_environment(env_name='academy_run_to_score', representation='simple115',
                                              logdir='./tmp1/football', write_video=True, render=True)
        state = env.reset()
        done = False
        total_reward = 0
        print('testing...')
        limit = 0
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            total_reward += reward
            state = state_

            limit += 1
            if limit > 50:
                break
        env.close()
        return total_reward

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)


import gfootball.env as football_env
import numpy as np

if __name__ == '__main__':
    image_based = False
    batch_size = 64
    if image_based:
        env = football_env.create_environment(env_name='academy_run_to_score', representation='pixels', render=False)
    else:
        env = football_env.create_environment(env_name='academy_run_to_score', representation='simple115',
                                              logdir='./tmp1/football', write_video=False, render=False)
    state_dims = env.observation_space.shape
    print(state_dims)
    n_actions = env.action_space.n
    # env = gym.make('LunarLander-v2')
    agent = Agent(lr=0.0005, gamma=0.99, n_actions=n_actions, epsilon=1.0,
                  batch_size=batch_size, input_dims=state_dims)
    n_games = 500
    scores = []
    eps_history = []
    best_reward = 0

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)

        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score: %.2f' % score, ' average score %.2f' % avg_score)

        i += 1
        if i < 100:
            continue
        avg_reward = np.mean([agent.evaluate() for _ in range(5)])
        print('total test reward=' + str(avg_reward))
        if avg_reward > best_reward:
            print('best reward=' + str(avg_reward))
            agent.save_model()
            best_reward = avg_reward
        if best_reward > 0.9 or i > n_games:
            target_reached = True

        env.reset()

    filename = 'lunarlander-dueling_ddqn.png'

    x = [i + 1 for i in range(n_games)]
    # plotLearning(x, ddqn_scores, eps_history, filename)
