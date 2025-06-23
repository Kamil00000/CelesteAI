🎮 Reinforcement Learning in Celeste Classic

Ten projekt wykorzystuje Deep Q-Network (DQN) do nauki agenta przechodzenia gry Celeste Classic, wymagającej platformówki opartej na precyzyjnych ruchach. Agent RL uczy się poruszać, skakać i unikać przeszkód, analizując zrzuty ekranu z gry bez dostępu do wewnętrznego stanu gry. Celem było sprawdzenie, czy agent może opanować podstawowe mechaniki tylko na podstawie danych wizyjnych.

Projekt łączy:

• głębokie uczenie (CNN),

• sterowanie klawiaturą przez pyautogui,

• przetwarzanie obrazu i system nagród,

• oraz uczenie poprzez eksplorację środowiska gry.


⚙️ Działanie programu

1. Inicjalizacja środowiska i agenta:
  
  • Monitorowanie określonego obszaru ekranu z grą.
  
  • Definicja 17 możliwych akcji (ruchy, skoki, dash).
  
  • Budowa sieci neuronowej (CNN + Dense) oraz target network.
  
  • Ustawienie epsilon = 1.0 (pełna eksploracja na starcie).

2. Trening agenta:
  
  • Reset gry i wykrywanie pozycji gracza (analiza kolorów).
  
  • Co krok: wybór akcji, jej wykonanie, obserwacja rezultatu, zapis doświadczenia.
  
  • System nagród bazuje na: ruchu, użyciu skoków/dash, śmierci, odkrywaniu nowych obszarów.

3. Uczenie się:
  
  • Replay buffer (10k doświadczeń), losowe próbkowanie batchy.
  
  • Obliczanie wartości docelowych Q.
  
  • Aktualizacja sieci co krok i synchronizacja co 10 epizodów.

4. Ewolucja agenta:

  • Z biegiem czasu: mniej losowych akcji, lepsze rozpoznawanie wzorców.

  • Od chaotycznego skakania do strategii z dash i precyzyjnym ruchem.

5. Testowanie:

• Po 500 epizodach treningu agent testowany jest w 10 rundach z minimalną eksploracją (ε = 0.01).
