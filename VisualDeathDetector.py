import threading
import time
from PIL import ImageGrab
import numpy as np
import cv2

class VisualDeathDetector:                              #x i y to współrzędne z game_region a nie z pulpitu
                                                    # x= 54, ystart =550, yend = 600 HP laptop
                                                    # x= 50, ystart =435,yend = 485 LG monitor
    def __init__(self, game_region, player_colors, x=54, y_start=550, y_end=600, color_tolerance=5, check_interval=0.05):
        self.game_region = game_region
        self.player_colors = player_colors
        self.color_tolerance = color_tolerance
        self.x = x
        self.y_start = y_start
        self.y_end = y_end
        self.check_interval = check_interval
        self.deaths = 0
        self.MAX_DEATHS = 3
        self._running = False
        self.thread = None
        self.death_detected = False

        self.last_mask_sum = 0  # suma pikseli koloru gracza z ostatniego pomiaru

        # Detekcja poziomu
       # self.previous_level_frame = None
       # self.level_changed = False
       # self.level_check_interval = 10.0  # sprawdzamy zmianę poziomu co X sekund
        #self.last_level_check_time = time.time()

       # self.menu_reference = cv2.imread("main_menu_reference.png")
       # if self.menu_reference is None:
       #     raise FileNotFoundError("Nie znaleziono pliku 'main_menu_reference.png'.")
        #self.menu_reference = cv2.resize(cv2.cvtColor(self.menu_reference, cv2.COLOR_BGR2GRAY), (64, 64))

    def start(self):
        self._running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self._running = False
        if self.thread:
            self.thread.join()

    def reset_death(self):
        self.death_detected = False
        self.last_mask_sum = 0


    def reset_level(self):
        self.level_changed = False
        self.previous_level_frame = None

    def was_death_detected(self):
        if self.death_detected:
            self.death_detected = False
            self.deaths += 1
            return True
        return False

   # def was_level_changed(self):
   #     if self.level_changed:
   #         self.level_changed = False
    #        return True
    #    return False

    #Porównanie obrazka z referencją ekranu menu głównego
    #def is_main_menu_screen(current_frame_gray, menu_reference_gray, threshold=100000):
        # Upewnij się, że obrazy mają ten sam rozmiar
    #    if current_frame_gray.shape != menu_reference_gray.shape:
    #        current_frame_gray = cv2.resize(current_frame_gray,
    #                                        (menu_reference_gray.shape[1], menu_reference_gray.shape[0]))

     #   diff = cv2.absdiff(current_frame_gray, menu_reference_gray)
    #    score = np.sum(diff)
    #    return score < threshold

        #Obliczanie różnicy w poziomach
    #def _is_significantly_different(self, img1, img2, threshold=1000):
    #    diff = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    #    return diff > threshold
        

    def _run(self):
        #Delay startu
        time.sleep(5.5)
        while self._running:
            screenshot = ImageGrab.grab(bbox=self.game_region)
            img_rgb = np.array(screenshot)

            vertical_strip = img_rgb[self.y_start:self.y_end + 1, self.x:self.x + 1]

            # Sprawdź czy vertical_strip nie jest pusty
            if vertical_strip.size == 0:
                print(
                    f"Pusty wycinek obrazu! Rozmiar obrazu: {img_rgb.shape}, wycinek x({self.x}-{self.x}), y({self.y_start}-{self.y_end})")
                time.sleep(self.check_interval)
                continue

            img_hsv = cv2.cvtColor(vertical_strip, cv2.COLOR_RGB2HSV)

            # Tworzymy maskę dla każdego koloru gracza
            masks = []
            for part in self.player_colors.values():
                lower = np.array(part['lower']) - self.color_tolerance
                upper = np.array(part['upper']) + self.color_tolerance
                masks.append(cv2.inRange(img_hsv, lower, upper))

            combined_mask = masks[0]
            for m in masks[1:]:
                combined_mask = cv2.bitwise_or(combined_mask, m)

            # Usuwamy szumy
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

            # Sumujemy liczbę pikseli pasujących do koloru gracza
            mask_sum = np.sum(combined_mask > 0)


            if self.last_mask_sum > 0 and mask_sum == 0:
                self.death_detected = True
                time.sleep(1.5)

                #Delay po śmierci
                if self.deaths >= self.MAX_DEATHS:
                    self.deaths = 0
                    time.sleep(5.5)

            self.last_mask_sum = mask_sum

            # =================================================================
            # DETEKCJA ZMIANY POZIOMU
            # =================================================================
            #now = time.time()
            #if now - self.last_level_check_time >= self.level_check_interval:
            #    resized_current = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY),
            #                                 (64, 64))  # mały rozmiar porównawczy

            #    if self.previous_level_frame is not None:
            #        if self._is_significantly_different(resized_current, self.previous_level_frame):
            #            # 💡 Tutaj sprawdzamy, czy to przypadkiem nie ekran menu
            #            if not self.is_main_menu_screen(img_rgb):
            #                self.level_changed = True
            #            else:
            #                print("Wykryto menu główne – ignoruję zmianę poziomu.")

            #    self.previous_level_frame = resized_current
             #   self.last_level_check_time = now

            time.sleep(self.check_interval)
