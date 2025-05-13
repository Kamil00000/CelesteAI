import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from celeste_env import CelesteEnv


class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.batch_size = 32
        self.train_start = 1000

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), strides=(2, 2), activation='relu',
                         input_shape=self.state_shape))
        model.add(Conv2D(64, (2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state[np.newaxis, ...], verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        for i in range(len(minibatch)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])

        self.model.fit(states, targets, batch_size=self.batch_size, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        # Zmieniamy rozszerzenie na .weights.h5
        if not name.endswith('.weights.h5'):
            if name.endswith('.h5'):
                name = name.replace('.h5', '.weights.h5')
            else:
                name += '.weights.h5'
        self.model.save_weights(name)

    def load(self, name):
        # Analogiczna zmiana dla wczytywania
        if not name.endswith('.weights.h5'):
            if name.endswith('.h5'):
                name = name.replace('.h5', '.weights.h5')
            else:
                name += '.weights.h5'
        self.model.load_weights(name)


def train_dqn(env, episodes=1000, render_every=50, save_every=100):
    state_shape = env.observation_shape
    action_size = env.action_size

    agent = DQNAgent(state_shape, action_size)
    done = False
    rewards_history = []
    epsilons_history = []
    best_reward = -np.inf

    models_dir = "dqn_models"
    os.makedirs(models_dir, exist_ok=True)

    for e in tqdm(range(episodes), desc="Training episodes"):
        state = env.reset()
        state = np.squeeze(state)
        total_reward = 0

        while True:
            action_idx = agent.act(state)
            action = env.action_space[action_idx]

            next_state, reward, done, info = env.step(action)
            next_state = np.squeeze(next_state)

            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.replay()

            if done:
                break

        if e % 10 == 0:
            agent.update_target_model()

        rewards_history.append(total_reward)
        epsilons_history.append(agent.epsilon)

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(f"{models_dir}/best_model.weights.h5")  # Zmienione rozszerzenie

        if e % save_every == 0:
            agent.save(f"{models_dir}/model_ep_{e}.weights.h5")  # Zmienione rozszerzenie

            # Zapis finalnego modelu
        agent.save(f"{models_dir}/final_model.weights.h5")  # Zmienione rozszerzenie

    np.savez(f"{env.output_dir}/training_data.npz",
             rewards=rewards_history,
             epsilons=epsilons_history)

    return agent, rewards_history


def plot_progress(output_dir, episode, rewards, epsilons):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Rewards per episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    plt.plot(epsilons)
    plt.title('Epsilon decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_progress_ep_{episode}.png")
    plt.close()


def test_agent(env, agent, episodes=10):
    agent.epsilon = 0.01

    test_rewards = []

    for e in range(episodes):
        state = env.reset()
        state = np.squeeze(state)
        total_reward = 0
        done = False

        while not done:
            action_idx = agent.act(state)
            action = env.action_space[action_idx]

            next_state, reward, done, info = env.step(action)
            next_state = np.squeeze(next_state)

            state = next_state
            total_reward += reward

        test_rewards.append(total_reward)
        print(f"Test episode {e + 1}/{episodes}, Reward: {total_reward:.2f}")

    return test_rewards


if __name__ == "__main__":
    env = CelesteEnv(output_dir="celeste_rl_output")

    try:
        trained_agent, rewards_history = train_dqn(env, episodes=500)

        print("\nTesting trained agent...")
        test_rewards = test_agent(env, trained_agent, episodes=10)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(rewards_history)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        plt.subplot(1, 2, 2)
        plt.plot(test_rewards, 'o-')
        plt.title('Test Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        plt.tight_layout()
        plt.savefig(f"{env.output_dir}/final_results.png")
        plt.show()

    finally:
        env.close()


if __name__ == "__main__":
    env = CelesteEnv(output_dir="celeste_test_output")
    try:
        trained_agent, _ = train_dqn(env, episodes=20)
    finally:
        env.close()