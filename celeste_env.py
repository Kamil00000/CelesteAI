import pyautogui
import numpy as np
import time
import cv2
from PIL import ImageGrab, Image, ImageDraw
from collections import deque
import matplotlib.pyplot as plt
import os


class CelesteEnv:
    def __init__(self, game_region=(215, 182, 730, 697), output_dir="debug_output"):
        self.game_region = game_region
        self.action_space = ['left', 'right', 'up', 'c', 'x', 'nothing']
        self.action_size = len(self.action_space)
        self.observation_shape = (35, 30, 1)

        self.debug = True
        self.output_dir = output_dir
        self.frame_count = 0
        self.episode_count = 0

        os.makedirs(self.output_dir, exist_ok=True)

        self.last_obs = None
        self.last_player_pos = None
        self.player_trajectory = deque(maxlen=100)
        self.steps_without_progress = 0
        self.max_steps_without_progress = 80  # zmniejszono dla szybszej reakcji
        self.total_steps = 0
        self.episode_reward = 0
        self.episode_rewards = []

        self.reward_weights = {
            'progress': 0.3,  # zwiększono nacisk na postęp
            'stuck': -2.0,    # bardziej karalne utknięcie
            'step': 0.01,     # mniejsza nagroda bazowa
            'backward': -0.2  # większa kara za cofanie się
        }

        self.player_colors = {
            'hair': {'lower': np.array([170, 230, 230]), 'upper': np.array([175, 255, 255]), 'debug_color': (255, 200, 90)},
            'skin': {'lower': np.array([12, 71, 230]), 'upper': np.array([14, 97, 255]), 'debug_color': (255, 165, 0)},
            'dress': {'lower': np.array([78, 230, 107]), 'upper': np.array([80, 255, 158]), 'debug_color': (0, 255, 0)},
        }

        self.min_player_size = 15
        self.color_tolerance = 5

    def _get_obs(self):
        screenshot = ImageGrab.grab(bbox=self.game_region)
        img_rgb = np.array(screenshot)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.observation_shape[0], self.observation_shape[1]))
        return np.expand_dims(resized, axis=-1), img_bgr

    def _detect_player(self, img_bgr):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        masks = []

        for part in self.player_colors:
            lower = np.clip(self.player_colors[part]['lower'] - self.color_tolerance, 0, 255)
            upper = np.clip(self.player_colors[part]['upper'] + self.color_tolerance, 0, 255)
            masks.append(cv2.inRange(hsv, lower, upper))

        combined_mask = cv2.bitwise_or(cv2.bitwise_or(masks[0], masks[1]), masks[2])
        filtered = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_player_size:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        return (x + w // 2, y + h // 2)

    def _save_debug_image(self, img_bgr, reward):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        debug_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(debug_img)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        for part, color in self.player_colors.items():
            lower = np.clip(color['lower'] - self.color_tolerance, 0, 255)
            upper = np.clip(color['upper'] + self.color_tolerance, 0, 255)
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) > 10:
                    pixels = [tuple(p[0]) for p in cnt]
                    if len(pixels) > 2:
                        draw.polygon(pixels, outline=color['debug_color'])

        if self.last_player_pos:
            x, y = self.last_player_pos
            draw.rectangle([x - 15, y - 25, x + 15, y + 25], outline='red', width=2)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill='magenta')

            if len(self.player_trajectory) > 1:
                prev_x, prev_y = self.player_trajectory[-2]
                draw.line([prev_x, prev_y, x, y], fill='white', width=2)

        draw.text((10, 10), f"Reward: {reward:.2f}", fill='white')
        draw.text((10, 30), f"Pos: {self.last_player_pos}", fill='white')

        debug_img.save(f"{self.output_dir}/debug_{self.frame_count:04d}.png")
        self.frame_count += 1

    def _compute_reward(self, img):
        reward = 0
        current_pos = self._detect_player(img)

        if current_pos and self.last_player_pos:
            # Silniejsza nagroda za ruch w prawo
            dx = current_pos[0] - self.last_player_pos[0]
            dy = self.last_player_pos[1] - current_pos[1]  # dodatnie gdy w górę

            # Główna nagroda za postęp w prawo (ważona nieliniowo)
            reward += np.sign(dx) * (dx ** 2) * 0.01  # kwadratowe wzmocnienie ruchu w prawo

            # Kara za cofanie się (bardziej agresywna)
            if dx < -2:
                reward -= 0.5

            # Dodatkowa nagroda za ruch w górę (np. pokonywanie przeszkód)
            reward += dy * 0.2

            self.player_trajectory.append(current_pos)

        # Kara za stagnację (bardziej progresywna)
        if abs(dx) < 1 and abs(dy) < 1:
            self.steps_without_progress += 1
            reward -= min(self.steps_without_progress / 10, 2.0)  # kara rośnie z czasem
        else:
            self.steps_without_progress = 0

        return reward

    def _detect_death(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) > 200

    def _press_key(self, action):
        if action == 'nothing':
            return
        pyautogui.keyDown(action)
        time.sleep(0.05 if action in ['left', 'right'] else 0.1)
        pyautogui.keyUp(action)

    def step(self, action):
        self._press_key(action)
        time.sleep(0.05)
        obs, img = self._get_obs()
        reward = self._compute_reward(img)
        self.episode_reward += reward

        done = False
        if self._detect_death(img) or self.total_steps >= 100:
            done = True

        if self.debug:
            self._save_debug_image(img, reward)

        info = {'player_pos': self.last_player_pos}
        self.last_obs = obs
        self.total_steps += 1
        return obs, reward, done, info

    def reset(self):
        pyautogui.click(button='left')
        time.sleep(1.0)
        pyautogui.press('x')
        time.sleep(2.0)

        self.last_obs, img = self._get_obs()
        self.last_player_pos = self._detect_player(img)
        self.player_trajectory.clear()
        self.steps_without_progress = 0
        self.total_steps = 0

        if self.episode_count > 0:
            self.episode_rewards.append(self.episode_reward)
            plt.figure(figsize=(10, 5))
            plt.plot(self.episode_rewards)
            plt.title("Nagrody za epizod")
            plt.xlabel("Epizod")
            plt.ylabel("Suma nagród")
            plt.grid()
            plt.savefig(f"{self.output_dir}/rewards_ep_{self.episode_count}.png")
            plt.close()

        self.episode_reward = 0
        self.episode_count += 1
        self.frame_count = 0
        return self.last_obs

    def close(self):
        if len(self.episode_rewards) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(self.episode_rewards)
            plt.title("Podsumowanie nagród")
            plt.xlabel("Epizod")
            plt.ylabel("Suma nagród")
            plt.grid()
            plt.savefig(f"{self.output_dir}/final_rewards.png")
            plt.close()


# Przykład użycia
if __name__ == "__main__":
    env = CelesteEnv(output_dir="celeste_debug")
    try:
        obs = env.reset()
        for _ in range(100):
            action = np.random.choice(env.action_space)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
    finally:
        env.close()
