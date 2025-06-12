import pyautogui
import numpy as np
import time
import cv2
from PIL import ImageGrab, Image, ImageDraw
from collections import deque
import matplotlib.pyplot as plt
import os
import random
#import sounddevice as sd
import numpy as np
#from scipy.signal import correlate

#from Death_detector import AudioDeathDetector
from VisualDeathDetector import VisualDeathDetector


class CelesteEnv:
    #  151, 241, 661, 751 HP laptop
    #  97, 208, 609, 720 LG monitor
    #  215, 182, 730, 697 Seba

    def __init__(self, game_region=(151, 241, 790, 880), output_dir="debug_output"):
        # Konfiguracja podstawowa
        self.game_region = game_region
        self.action_space = self._create_action_combinations()
        self.action_size = len(self.action_space)
        self.observation_shape = (84, 84, 1)  # Skala szarości

        # Mapowanie klawiszy Celeste Classic
        self.key_mapping = {
            'left': 'left',
            'right': 'right',
            'up': 'up',
            'jump': 'c',  # Skok w Celeste Classic
            'dash': 'x',  # Dash w Celeste Classic
        #    'long_jump': 'c'
        }

        # Debugowanie
        self.debug = True
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Stan środowiska
        self.last_obs = None
        self.last_player_pos = None
        self.player_trajectory = deque(maxlen=100)
        self.steps_on_spawn = 0
        self.visited = set() #Do sprawdzania odwiedzonych kafelków w grze
        self.steps_without_progress = 0
        self.max_steps_without_progress = 150
        self.total_steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.episode_count = 0
        self.max_episode_steps = 150

        # System nagród

        self.reward_weights = {
            'right': 0.015,
            'right_nonlinear': 0.0002,
            'up': 0.015,
            'stuck': -0.4,
            'death': -5.0,
            'time': -0.0002,
            'dash_used': 0.05,  #najlepsze dotąd były 0.08   #OSTATNIE 0.1
            'jump_used': 0.08,  #najlepsze dotąd były 0.05   #OSTATNIE 0.08
            'max_right_progress': 0.5, #najlepsze dotąd były 0.04  #OSTATNIE 0.5
            'max_top_progress': 0.9, #najlepsze dotąd były 0.9  #OSTATNIE 1.1
            'blank_dash': -1.0,
            'combo_dash_jump': 0.8, #najlepsze dotąd były 0.8 #OSTATNIE 1.4
            'backward': -0.04,   #OSTATNIE -0.03
            'spawnCamp': -0.02,
            'new_tile': 0.2,
            'next_level': 10.0
        }


        # Detekcja gracza
        self.player_colors = {
            'hair': {'lower': [170, 230, 230], 'upper': [175, 255, 255]},
            'skin': {'lower': [12, 71, 230], 'upper': [14, 97, 255]},
            'dress': {'lower': [78, 230, 107], 'upper': [80, 255, 158]}
        }
        self.min_player_size = 60
        self.color_tolerance = 5

        # Tworzenie instancji detektora śmierci
        #self.audio_detector = AudioDeathDetector()
        #self.audio_detector.start()

        self.deaths = 0
        self.MAX_DEATHS = 3

        self.visual_detector = VisualDeathDetector(
            self.game_region,
            player_colors=self.player_colors
        )
        self.visual_detector.start()


        #Przmieszczanie się w strone mety
        #Wartości początkowe to jest spawn
        self.max_x_reached = 50
        self.min_y_reached = 400

        #Zmienna dla info o zmianie poziomu (nie działa)
        #self.level_number = 1


    def _create_action_combinations(self):
        return [
            ['right'],  # Prawo
            ['right', 'jump'],  # Skok w prawo
            ['right', 'dash'],  # Dash w prawo
            ['left'],  # Lewo
            ['left', 'jump'],  # Skok w lewo
            ['left', 'dash'],  # Dash w lewo
            ['up', 'jump'],  # Skok w górę
            ['up', 'dash'],  # Dash w górę
            ['jump'],  # Sam skok
            ['dash'],  # Sam dash
            ['right', 'up', 'jump'],
            ['left', 'up', 'jump'],
            ['right', 'up', 'dash'],
            ['left', 'up', 'dash'],
            ['right', 'up', 'jump', 'dash'],
            ['left', 'up', 'jump', 'dash'],
            ['nothing']  # Brak akcji
        ]

    def _get_obs(self):
        screenshot = ImageGrab.grab(bbox=self.game_region)
        img_rgb = np.array(screenshot)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(img_gray, (self.observation_shape[0], self.observation_shape[1]))
        normalized = resized / 255.0
        return np.expand_dims(normalized, axis=-1), img_rgb

    def _detect_player(self, img_rgb):
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        masks = []

        for part in self.player_colors.values():
            lower = np.array(part['lower']) - self.color_tolerance
            upper = np.array(part['upper']) + self.color_tolerance
            masks.append(cv2.inRange(img_hsv, lower, upper))

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

    def _compute_reward(self, current_pos, action):
        reward = 0
        dx, dy = 0, 0

        if current_pos and self.last_player_pos:
            dx = current_pos[0] - self.last_player_pos[0]
            dy = self.last_player_pos[1] - current_pos[1]  # Dodatnie gdy w górę

            # Nagrody za ruch
            if dx > 5:
                reward += self.reward_weights['right'] * dx

            if dx > 5 and dy > 5:
                reward += self.reward_weights['right_nonlinear'] * (dx ** 2)

            #kara za cofanie staje się nagrodą po przekroczeniu 220 pixela w osi y żeby zachęcić do skakania na platforme po lewej
            if dx < 5 and current_pos[1] < 220:
                reward -= self.reward_weights['backward'] * abs(dx)  # nagroda
            elif dx < 5:
                reward += self.reward_weights['backward'] * abs(dx) # kara


            # Kara za stagnację
            if dx < 5 and dy < 5:
                self.steps_without_progress += 1
                reward += self.reward_weights['stuck'] * self.steps_without_progress
            else:
                self.steps_without_progress = 0

            #Kara za kiszenie sie na spawnie
            if current_pos[0] < 100:
                self.steps_on_spawn += 1
                reward += self.reward_weights['spawnCamp'] * self.steps_on_spawn
            else:
                self.steps_on_spawn = 0

            #Nagroda za pobicie rekordu
            if current_pos[0] > self.max_x_reached:
                reward += self.reward_weights['max_right_progress']
                self.max_x_reached = current_pos[0]

            if current_pos[1] < self.min_y_reached:
                reward += self.reward_weights['max_top_progress']
                self.min_y_reached = current_pos[1]

            tile = (current_pos[0] // 32, current_pos[1] // 32)  # zaokrąglanie
            if tile not in self.visited:
                reward += self.reward_weights['new_tile'] # nagroda za eksplorację
                self.visited.add(tile)

            # Nagrody za specjalne akcje
            if 'dash' in action and (dx < -5 or dx > 5) and dy > 5 and current_pos[0] > 100:
                reward += self.reward_weights['dash_used']
            if 'dash' in action and (-5 <= dx <= 5) and (-5 <= dy <= 5):
                reward += self.reward_weights['blank_dash']
            if 'jump' in action and (dx < -5 or dx > 5) and dy > 5 and current_pos[0] > 100:
                reward += self.reward_weights['jump_used']

            if 'dash' in action and 'jump' in action and ((dx < -5 or dx > 5) and dy > 5 and current_pos[0] > 100):
                reward += self.reward_weights['combo_dash_jump']  #nagroda za złożoną akcję


        # Kara za czas
        reward += self.reward_weights['time']

        return reward, dx, dy

    def _press_action(self, action):


        # Zwolnienie wszystkich klawiszy
        for key in ['left', 'right', 'up', 'jump', 'dash']:
            pyautogui.keyUp(self.key_mapping[key])

        # Naciśnięcie wymaganych klawiszy
        for action_part in action:
            if action_part != 'nothing':
                pyautogui.keyDown(self.key_mapping[action_part])

        # Specjalne czasy dla różnych akcji
        #if 'long_jump' in action:
        #    time.sleep(0.6)
        if 'jump' in action:
            time.sleep(0.6)    #pierwotnie 0.08 czas wydłużony żeby robił dłuższe skoki
        elif 'dash' in action:
            time.sleep(0.3)  # Dłuższy czas dla dash pierwotnie 0.12
        else:
            time.sleep(0.1)  # Krótki czas dla ruchu pierwotnie 0.05

        # Zwolnienie klawiszy
        for action_part in action:
            if action_part != 'nothing':
                pyautogui.keyUp(self.key_mapping[action_part])


    def step(self, action):
        self._press_action(action)
        time.sleep(0.03)  # Czas na reakcję gry

        obs, img_rgb = self._get_obs()
        current_pos = self._detect_player(img_rgb)
        reward, dx, dy = self._compute_reward(current_pos, action)
        self.episode_reward += reward

        done = False
        death = self.visual_detector.was_death_detected()

        if death:
            self.deaths += 1
            reward += self.reward_weights['death']
            self.visual_detector.reset_death()
            if self.deaths >= self.MAX_DEATHS:
                done = True
        elif self.total_steps >= self.max_episode_steps:
            done = True

        #new_level = self.visual_detector.was_level_changed()
        #if new_level:
        #    self.level_number += 1
        #    reward += self.reward_weights['next_level']
        #    self.visual_detector.reset_level()
        #    self.reset_personal_records()
        #    print(f"nowy poziom!!!")


        if self.debug and self.total_steps % 10 == 0:
            self._save_debug_image(img_rgb, reward, dx, dy, action)

        info = {
            'player_pos': current_pos,
            'movement': (dx, dy),
            'death': self.deaths,
            #'level num':  self.level_number,
            'action': action
        }

        self.last_obs = obs
        self.last_player_pos = current_pos
        self.total_steps += 1

        if current_pos:
            self.player_trajectory.append(current_pos)

        return obs, reward, done, info

    def _save_debug_image(self, img_rgb, reward, dx, dy, action):
        debug_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(debug_img)

        # Rysowanie trajektorii
        if len(self.player_trajectory) > 1:
            for i in range(1, len(self.player_trajectory)):
                draw.line([*self.player_trajectory[i - 1], *self.player_trajectory[i]],
                          fill=(255, 255, 0), width=2)

        # Dodawanie informacji
        draw.text((10, 10), f"Reward: {reward:.2f}", fill=(0, 0, 0))
        draw.text((10, 30), f"Pos: {self.last_player_pos}", fill=(0, 0, 0))
        draw.text((10, 50), f"Move: dx={dx}, dy={dy}", fill=(0, 0, 0))
        draw.text((10, 70), f"Action: {action}", fill=(0, 0, 0))
        draw.text((10, 90), f"Steps: {self.total_steps}", fill=(0, 0, 0))

        # Rysowanie prostokąta oznaczającego gracza
        player_pos = self._detect_player(img_rgb)

        if player_pos is None and self.last_player_pos is not None:
            # Handle the case where player isn't detected
            player_x, player_y = self.last_player_pos  # or some default values
        elif player_pos is None and self.last_player_pos is None:
            player_x, player_y = 0, 0
        else:
            player_x, player_y = player_pos

        player_size = 30  # Możesz dostosować rozmiar prostokąta do swojego gracza
        rect_x1 = player_x - player_size // 2
        rect_y1 = player_y - player_size // 2
        rect_x2 = player_x + player_size // 2
        rect_y2 = player_y + player_size // 2

        # Rysowanie prostokąta
        draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], outline="red", width=2)

        debug_img.save(f"{self.output_dir}/ep_{self.episode_count}_step_{self.total_steps:04d}.png")

    def reset(self):
        # Resetowanie gry
        #pyautogui.click(button='left')
        #Współrzędne przycisku reset żeby nie musieć trzymać na nim myszki
        #HP laptop 182, 912
        #LG monitor 123, 744
        pyautogui.click(x=182, y=912)
        time.sleep(0.2)
        pyautogui.keyDown('x')
        time.sleep(0.2)
        pyautogui.keyUp('x')
        time.sleep(2.9)

        # Reset stanu środowiska
        self.last_obs, img_rgb = self._get_obs()
        self.last_player_pos = self._detect_player(img_rgb)
        self.player_trajectory.clear()
        self.steps_without_progress = 0
        self.steps_on_spawn = 0
        self.total_steps = 0
        self.visited.clear()
        self.deaths = 0
        #self.level_number = 1
        # Zapis wyników jeśli to nie pierwszy epizod
        if self.episode_count > 0:
            self.episode_rewards.append(self.episode_reward)
            self._plot_training_progress()

        self.episode_reward = 0
        self.episode_count += 1
        self.max_x_reached = 50
        self.min_y_reached = 400

        return self.last_obs

    #def reset_personal_records(self):
    #    self.steps_without_progress = 0
    #    self.steps_on_spawn = 0
    #    self.visited.clear()
    #    self.max_x_reached = 50
    #    self.min_y_reached = 400

    def _plot_training_progress(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.episode_rewards)
        plt.title(f"Training Progress (Last 100 avg: {np.mean(self.episode_rewards[-100:]):.2f})")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid()
        plt.savefig(f"{self.output_dir}/training_progress.png")
        plt.close()

    def close(self):
        if len(self.episode_rewards) > 0:
            self._plot_training_progress()
        self.visual_detector.stop()
