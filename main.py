import numpy as np
import random
import pickle
from celeste_env import CelesteEnv

env = CelesteEnv()
n_actions = len(env.action_space)
q_table = {}

def get_state(obs):
    # Bardzo prymitywne uproszczenie: tylko średnia jasność
    avg_brightness = int(np.mean(obs) // 10)
    return avg_brightness

# Hiperparametry
episodes = 50
learning_rate = 0.1
discount = 0.95
epsilon = 1.0
epsilon_decay = 0.95
min_epsilon = 0.1

for episode in range(episodes):
    obs = env.reset()
    state = get_state(obs)
    #env.debug = True  # Włącz tryb debugowania
    total_reward = 0

    for step in range(200):
        if random.random() < epsilon:
            action_idx = random.randint(0, n_actions - 1)
        else:
            if state not in q_table:
                q_table[state] = np.zeros(n_actions)
            action_idx = np.argmax(q_table[state])

        action = env.action_space[action_idx]
        next_obs, reward, done, _ = env.step(action)
        next_state = get_state(next_obs)

        if state not in q_table:
            q_table[state] = np.zeros(n_actions)
        if next_state not in q_table:
            q_table[next_state] = np.zeros(n_actions)

        old_value = q_table[state][action_idx]
        next_max = np.max(q_table[next_state])

        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount * next_max)
        q_table[state][action_idx] = new_value

        state = next_state
        total_reward += reward

        print(f"Episode {episode}, Step {step}, Action {action}, Reward {reward:.4f}, Total {total_reward:.4f}")

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Zapisz wytrenowaną tablicę Q
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)
