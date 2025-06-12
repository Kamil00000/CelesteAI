import time
import pyautogui
#import sounddevice as sd

#from Death_detector import AudioDeathDetector6

while(True):
    print(pyautogui.position())  # Zwróci aktualną pozycję kursora
    time.sleep(1)


#import soundfile as sf

#data, samplerate = sf.read("death.wav") # Sprawdzenie częstotliwości próbkowania dźwięku
#(f"Sample rate: {samplerate} Hz")



# Wybór urządzenia audio
#print(sd.query_devices())
#sd.default.device = [None, 2]  # [output, input]

#testy odczytu dcźwięków

#audio_detector = AudioDeathDetector6()
#audio_detector.start()
#while(True):
    #pass




#Delay został praktycznie zniwelowany teraz wyłącznie dorobić działające parametry albo dograć nagranie na konkretnej
#Głośności dźwięku bo się pierdoli
#Można też spróbować do innego sterownika dźwięku podpiąć program


#Stanęło na tym że audio jednak chujowo działa spróbować z odzszumioną wersją każdego z przykładów
#Odszumić jeszcze raz jak nie zadziała to robić coś innego