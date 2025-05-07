import pyautogui
import numpy as np
import time
import cv2
from PIL import ImageGrab, Image, ImageDraw
from collections import deque
import matplotlib.pyplot as plt
import os
from datetime import datetime


class CelesteEnv:
    def __init__(self, game_region=(215, 182, 730, 697), output_dir="debug_output"):
        """
        Środowisko do nauki RL dla Celeste Classic z zapisem debug do plików
        """
        # Parametry środowiska
        self.game_region = game_region
        self.action_space = ['left', 'right', 'up', 'c', 'x', 'nothing']
        self.action_size = len(self.action_space)
        self.observation_shape = (35, 30, 1)

        # Konfiguracja debugowania do plików
        self.debug = True
        self.output_dir = output_dir
        self.frame_count = 0
        self.episode_count = 0

        # Utwórz katalog wyjściowy jeśli nie istnieje
        os.makedirs(self.output_dir, exist_ok=True)

        # Śledzenie stanu
        self.last_obs = None
        self.last_player_pos = None
        self.player_trajectory = deque(maxlen=100)
        self.steps_without_progress = 0
        self.max_steps_without_progress = 100
        self.total_steps = 0
        self.episode_reward = 0
        self.episode_rewards = []

        # Parametry nagród
        self.reward_weights = {
            'progress': 0.2,
            'stuck': -1.0,
            'step': 0.02,
            'backward': -0.1
        }

        # Kolory HSV dla Celeste Classic
        self.player_colors = {
            'hair': {
                'lower': np.array([170, 230, 230]),  # H:341/2=170, S:90*2.55=230, V:90*2.55=230
                'upper': np.array([175, 255, 255]),  # H:351/2=175, S:110*2.55=280->255, V:110*2.55=280->255
                'debug_color': (255, 200, 90)
            },
            'skin': {
                'lower': np.array([12, 71, 230]),  # H:24/2=12, S:28*2.55=71, V:90*2.55=230
                'upper': np.array([14, 97, 255]),  # H:29/2=14.5->14, S:38*2.55=97, V:110*2.55=280->255
                'debug_color': (255, 165, 0)
            },
            'dress': {
                'lower': np.array([78, 230, 107]),  # H:156/2=78, S:90*2.55=230, V:42*2.55=107
                'upper': np.array([80, 255, 158]),  # H:161/2=80, S:110*2.55=280->255, V:62*2.55=158
                'debug_color': (0, 255, 0)
            }
        }

        self.min_player_size = 15
        self.color_tolerance = 5  # Mniejsza tolerancja dla dokładności

    def _get_obs(self):
        screenshot = ImageGrab.grab(bbox=self.game_region)
        img_rgb = np.array(screenshot)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.observation_shape[0], self.observation_shape[1]))
        return np.expand_dims(resized, axis=-1), img_bgr

    def _detect_player(self, img_bgr):
        """Dokładna detekcja HSV z Twoimi wartościami"""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        masks = []
        for part in ['hair', 'skin', 'dress']:
            lower = np.clip(self.player_colors[part]['lower'] - self.color_tolerance, [0, 0, 0], [179, 255, 255])
            upper = np.clip(self.player_colors[part]['upper'] + self.color_tolerance, [0, 0, 0], [179, 255, 255])
            masks.append(cv2.inRange(hsv, lower, upper))

        combined_mask = cv2.bitwise_or(masks[0], masks[1])
        combined_mask = cv2.bitwise_or(combined_mask, masks[2])

        kernel = np.ones((3, 3), np.uint8)
        filtered = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_player_size:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        return (x + w // 2, y + h // 2)

    def _save_debug_image(self, img_bgr, reward):
        """Zachowaj poprawne kolory używając PIL"""
        # Konwersja BGR do RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        debug_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(debug_img)

        # Zaznaczanie części postaci
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        for part, color in self.player_colors.items():
            lower = np.clip(color['lower'] - self.color_tolerance, [0, 0, 0], [179, 255, 255])
            upper = np.clip(color['upper'] + self.color_tolerance, [0, 0, 0], [179, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)

            # Zaznacz tylko brzegi wykrytych obszarów
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 10:  # Ignoruj małe obszary
                    pixels = []
                    for point in cnt:
                        pixels.append(tuple(point[0]))
                    if len(pixels) > 2:
                        draw.polygon(pixels, outline=color['debug_color'])

        # Zaznacz pozycję gracza
        if self.last_player_pos:
            x, y = self.last_player_pos
            draw.rectangle([x - 15, y - 25, x + 15, y + 25], outline='red', width=2)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill='magenta')

            if len(self.player_trajectory) > 1:
                prev_x, prev_y = self.player_trajectory[-2]
                draw.line([prev_x, prev_y, x, y], fill='white', width=2)

        # Dodaj informacje tekstowe
        draw.text((10, 10), f"Reward: {reward:.2f}", fill='white')
        draw.text((10, 30), f"Pos: {self.last_player_pos}", fill='white')

        # Zapisz
        debug_img.save(f"{self.output_dir}/debug_{self.frame_count:04d}.png")
        self.frame_count += 1

    def _compute_reward(self, img):
        """Obliczanie nagrody"""
        reward = 0
        current_player_pos = self._detect_player(img)

        if current_player_pos and self.last_player_pos:
            progress_x = current_player_pos[0] - self.last_player_pos[0]
            progress_y = self.last_player_pos[1] - current_player_pos[1]

            reward += (0.8 * progress_x + 0.2 * progress_y) * self.reward_weights['progress']

            if progress_x < -2:
                reward += self.reward_weights['backward']

            self.player_trajectory.append(current_player_pos)

        if abs(reward) < 0.05:
            self.steps_without_progress += 1
            if self.steps_without_progress > self.max_steps_without_progress:
                reward += self.reward_weights['stuck']
        else:
            self.steps_without_progress = 0

        reward += self.reward_weights['step']
        self.last_player_pos = current_player_pos
        return reward

    def _detect_death(self, img):
        """Wykrywanie śmierci po białym błysku"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) > 200

    def _press_key(self, action):
        """Symulacja naciśnięcia klawisza"""
        if action == 'nothing':
            return

        pyautogui.keyDown(action)
        time.sleep(0.05 if action in ['left', 'right'] else 0.1)
        pyautogui.keyUp(action)

    def step(self, action):
        """Wykonanie kroku środowiska"""
        self._press_key(action)
        time.sleep(0.05)

        obs, img = self._get_obs()
        reward = self._compute_reward(img)
        self.episode_reward += reward

        done = False
        if self._detect_death(img):
            done = True
        elif self.total_steps >= 2000:
            done = True

        info = {'player_pos': self.last_player_pos}
        self.last_obs = obs
        self.total_steps += 1

        if self.debug:
            self._save_debug_image(img, reward)

        return obs, reward, done, info

    def reset(self):
        """Reset środowiska"""
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

            # Zapisz wykres nagród po każdym epizodzie
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
        """Zamykanie środowiska"""
        # Zapisanie finalnego wykresu nagród
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