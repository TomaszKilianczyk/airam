print("\n--- Instalacja zakończona. Próbuję załadować moduły... ---")

# --- 2. Importy ---
import re
import os
import pickle
import webbrowser
import threading
import time
import logging

# --- DODANE IMPORTOVANIA DLA OUPENAI API ---
import os
import openai # Now using openai library
# --- KONIEC DODANYCH IMPORTÓW ---

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
# from transformers import AutoTokenizer, AutoModelForCausalLM #
import torch


# --- 3. Klasy pamięci ---
# ... (Twoje klasy L1Memory, L2Memory, L3Memory, ACRMechanism pozostają bez zmian) ...
# Pamiętaj, że w klasie ACRMechanism masz `self.last_injected_line` itp. to też będą do wyciągnięcia do HTML
class L1Memory:
    def __init__(self, capacity=5):  # Dodaj pojemność L1
        self.capacity = capacity
        self.contents = []  # Zmieniamy na listę, łatwiejszą do zarządzania FIFO

    def add(self, item):
        if len(self.contents) >= self.capacity:
            self.contents.pop(0)  # Usuń najstarszy
        self.contents.append(item)

    def get_all_contents(self):
        return self.contents

    def clear(self):
        self.contents = []


class L2Memory:
    def __init__(self, slot_count):
        self.slots = [{"id": None, "content": None} for _ in range(slot_count)]
        self.activity_scores = [0.0] * slot_count
        # --- NOWE: Dodane zmienne do śledzenia ostatniej aktywacji L2 ---
        self.last_activated_content = None
        self.last_activated_sim = None
        # --- KONIEC NOWE ---

    def add_to_slot(self, idx, element, iteration):
        # Upewniamy się, że element jest słownikiem z 'id' i 'content'
        if not isinstance(element, dict) or 'id' not in element or 'content' not in element:
            raise ValueError("Element must be a dictionary with 'id' and 'content' keys.")
        self.slots[idx] = element
        self.activity_scores[idx] = 1.0  # Aktywacja slotu
        # Resetuj śledzenie aktywacji L2, gdy slot jest wypełniany nową treścią
        self.last_activated_content = None
        self.last_activated_sim = None

    def get_all_contents(self):
        return [s['content'] for s in self.slots if s['content']]  # Filtruj puste sloty

    def decay_and_consolidate_l2(self, decay_rate=0.95):  # Dodany parametr
        for i in range(len(self.activity_scores)):
            if self.slots[i]['content']:  # Tylko jeśli slot jest zajęty
                self.activity_scores[i] *= decay_rate

    def query_l3_and_activate_l2(self, query_emb, encoder, memory_top_k=3):  # Dodany parametr
        # --- NOWE: Resetuj śledzenie aktywacji L2 przed nowym zapytaniem ---
        self.last_activated_content = None
        self.last_activated_sim = None
        # --- KONIEC NOWE ---

        texts = self.get_all_contents()
        if not texts or query_emb is None:
            return

        emb = encoder.encode(texts, convert_to_tensor=True).cpu()
        query_emb_cpu = query_emb.cpu()  # Upewnij się, że query_emb też jest na CPU

        sims = util.cos_sim(query_emb_cpu, emb)[0]
        # Sortuj indeksy podobieństw, żeby aktywować najbardziej relewantne
        for i in sims.argsort(descending=True)[:memory_top_k]:
            if sims[i].item() > 0.4:
                # Znajdź indeks oryginalnego slotu w L2, odpowiadający temu tekstowi
                original_slot_idx = -1
                for slot_idx, slot in enumerate(self.slots):
                    if slot['content'] == texts[i.item()]:
                        original_slot_idx = slot_idx
                        break

                if original_slot_idx != -1:
                    self.activity_scores[original_slot_idx] += 0.5  # Aktywacja
                    # --- NOWE: Zapisz ostatnio aktywowaną treść i podobieństwo ---
                    self.last_activated_content = texts[i.item()]
                    self.last_activated_sim = sims[i].item()
                    # --- KONIEC NOWE ---
                    app.logger.info(
                        f"Aktywowano slot L2 {original_slot_idx} z treścią '{texts[i.item()]}' (podobieństwo: {sims[i].item():.2f}).")
                    break  # Aktywujemy tylko jeden najbardziej podobny, aby śledzić 'last_activated' łatwiej
                else:
                    app.logger.warning(f"Nie znaleziono oryginalnego slotu dla tekstu: '{texts[i.item()]}'")


class L3Memory:
    def __init__(self, encoder):
        self.entries = {}
        self.encoder = encoder

    def add(self, id, content):
        self.entries[id] = content

    def save_to_file(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.entries, f)

    def load_from_file(self, path):
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.entries = pickle.load(f)
                app.logger.info(f"Załadowano {len(self.entries)} wpisów z L3.")
            except (EOFError, pickle.UnpicklingError, FileNotFoundError):
                app.logger.warning("Błąd ładowania pamięci L3 lub plik nie istnieje, tworzę nową pamięć L3.")
                self.entries = {}
        else:
            app.logger.info("Plik L3 nie istnieje, tworzę nową pamięć L3.")
            self.entries = {}


class ACRMechanism:
    def __init__(self, l2, encoder, file_path="plik.txt", cache_path="plik_embeddings_cache.pkl"):
        self.l2 = l2
        self.encoder = encoder
        self.file_path = file_path
        self.cache_path = cache_path
        self.promoted = set()
        self.lines = []
        self.embeddings = None

        self.last_activated_content = None
        self.last_activated_sim = None
        self.last_injected_line = None
        self.last_injected_slot_idx = None

        self._load_file_and_cache()

    def _load_file_and_cache(self):
        if not os.path.exists(self.file_path):
            open(self.file_path, "w", encoding="utf-8").close()
            app.logger.info(f"Plik L2 ({self.file_path}) nie istnieje, tworzę pusty.")
            self.lines = []
            self.embeddings = None
            return

        with open(self.file_path, "r", encoding="utf-8") as f:
            self.lines = [l.strip() for l in f if l.strip()]

        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    cache = pickle.load(f)
                if cache.get("lines") == self.lines:
                    self.embeddings = cache["embeddings"]
                    app.logger.info(f"Załadowano {len(self.embeddings)} embeddings z cache L2.")
                    return
            except (EOFError, pickle.UnpicklingError, FileNotFoundError):
                app.logger.warning("Błąd ładowania cache L2 lub plik nie istnieje, odbuduję embeddings.")
                self.embeddings = None

        if self.lines:
            app.logger.info("Rebuduję embeddings dla L2...")
            self.embeddings = self.encoder.encode(self.lines, convert_to_tensor=True).cpu()
            with open(self.cache_path, "wb") as f:
                pickle.dump({"lines": self.lines, "embeddings": self.embeddings}, f)
            app.logger.info("Embeddings L2 odbudowane i zapisane do cache.")
        else:
            self.embeddings = None
            app.logger.info("Brak linii w L2 do odbudowania embeddings.")

    def reload_l2_from_file(self):
        app.logger.info(f"Przeładowuję plik L2: {self.file_path} i odbudowuję embeddingi...")
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
            app.logger.info(f"Usunięto stary cache L2: {self.cache_path}")
        self._load_file_and_cache()

    def inject_relevant_to_l2(self, iteration, query_emb, memory_top_k=None, l2_sim_threshold=0.4,
                              exact_match_threshold=0.75, injection_limit=3):
        # memory_top_k jest teraz opcjonalne i nie używane do wstępnego filtrowania
        # l2_sim_threshold (dolny próg) ustawiony na 0.4
        # exact_match_threshold (górny próg) ustawiony na 0.75
        # injection_limit - ile max informacji wstrzyknąć

        self.last_injected_line = None  # Będziemy śledzić tylko ostatnią, lub zmienić na listę
        self.last_injected_slot_idx = None  # To samo

        injected_count = 0  # Licznik wstrzykniętych informacji

        if not self.lines or self.embeddings is None or query_emb is None:
            # Ta sekcja dummy content nie jest już potrzebna, jeśli chcemy wstrzykiwać tylko relewantne fakty.
            # Jeśli ją zostawisz, będzie wstrzykiwać dummy content nawet gdy są linie, ale brak pasujących.
            # app.logger.info(f"Brak linii w pliku lub embeddings. Nie wstrzyknięto nic.")
            # return
            # Jeśli faktycznie nie ma linii, to możemy zalogować i zakończyć.
            app.logger.info(f"Brak linii w pliku lub embeddings do wstrzyknięcia. Kontynuuję bez wstrzykiwania.")
            return

        query_emb_cpu = query_emb.cpu()
        sims = util.cos_sim(query_emb_cpu, self.embeddings)[0]

        # Iterujemy przez WSZYSTKIE posortowane podobieństwa
        for i in sims.argsort(descending=True):
            # Sprawdzamy limit wstrzykniętych informacji na początku każdej iteracji
            if injected_count >= injection_limit:
                app.logger.info(f"Osiągnięto limit wstrzykniętych informacji ({injection_limit}). Przerywam pętlę.")
                break  # Przerywamy pętlę, jeśli wstrzyknęliśmy już wystarczająco dużo

            current_sim = sims[i].item()
            line = self.lines[i.item()]

            normalized_line = re.sub(r'\s+', ' ', line.lower()).strip()

            # --- FILTROWANIE NUMER 1: Podobieństwo semantyczne (za wysokie lub za niskie) ---
            if current_sim < l2_sim_threshold:
                app.logger.info(
                    f"Pomijam wstrzyknięcie '{line}' do L2: podobieństwo ({current_sim:.2f}) jest zbyt niskie (poniżej {l2_sim_threshold}).")
                # Ponieważ lista jest posortowana malejąco, jeśli podobieństwo jest poniżej progu,
                # to wszystkie kolejne będą również poniżej. Możemy tu bezpiecznie przerwać.
                break  # Zmieniono continue na break dla efektywności

            if current_sim > exact_match_threshold:
                app.logger.info(
                    f"Pomijam wstrzyknięcie '{line}' do L2: podobieństwo ({current_sim:.2f}) jest zbyt wysokie (powyżej {exact_match_threshold}), uznano za powtórzenie pytania lub zbyt ogólny wpis.")
                continue

            # --- FILTROWANIE NUMER 2: Reguły tekstowe (wykrywanie pytań, powitań, błędów, niefaktów) ---
            is_dialog_prefix = line.startswith("Użytkownik:") or line.startswith("AIram:")

            is_question_by_keywords = False
            question_keywords_start_regex = r"^\s*(ile|co|kto|gdzie|kiedy|jak|dlaczego|czy|po co|jaka|jakie|jaki|który)\b"

            if "?" in line:
                is_question_by_keywords = True
            elif re.search(question_keywords_start_regex, normalized_line):
                if not re.search(r"\b(jest|wynosi|ma|to|się|posiada|liczy)\b", normalized_line):
                    is_question_by_keywords = True

            is_ai_greeting_or_error = re.match(
                r"^[Aa][Ii]ram:\s*(Cześć!|Jak mogę pomóc\?|Nie wiem\.|Przepraszam, wystąpił błąd\.|Wystąpił błąd podczas komunikacji\.|Nie mam informacji o.*)",
                line) is not None
            is_prompt_fragment = "pytanie użytkownika:" in normalized_line or \
                                 "kontekst z pamięci długoterminowej:" in normalized_line or \
                                 "historia rozmowy:" in normalized_line or \
                                 "ai ram:" in normalized_line

            is_too_short_and_irrelevant = len(
                line.split()) <= 2 and not is_dialog_prefix and not line.isdigit() and not any(
                unit in normalized_line for unit in ["cm", "metr", "m", "kg", "l"])

            is_ai_dont_know = "nie wiem" in normalized_line and not (
                    "informacj" in normalized_line and "o" in normalized_line and "na temat" in normalized_line or
                    "mam dostępu do" in normalized_line or
                    "postaram się pomóc" in normalized_line or
                    "wiem, że" in normalized_line or
                    re.search(
                        r"nie wiem(,|\.)?\s*(nie mam)?\s*(dostępu)?\s*(informacj)?\s*(o)?\s*(nagród)?\s*(pami)?\s*(lub działa)?",
                        normalized_line) is not None
            )

            if is_dialog_prefix or is_question_by_keywords or is_ai_greeting_or_error or is_prompt_fragment or is_too_short_and_irrelevant or is_ai_dont_know:
                app.logger.info(
                    f"Pomijam wstrzyknięcie '{line}' do L2: uznano za niefakt/pytanie/dialog/za krótkie na podstawie wzorców tekstowych.")
                continue

            # --- FILTROWANIE NUMER 3: Czy linia została już wcześniej promowana do L2 ---
            lid = f"file_{hash(line)}"
            if lid in self.promoted:
                app.logger.info(f"Pomijam wstrzyknięcie '{line}' do L2: już wcześniej promowane.")
                continue

            # Jeśli linia przeszła WSZYSTKIE filtry, to jest kandydatem do wstrzyknięcia
            # Znajdź najmniej aktywny slot w L2
            min_score_idx = self.l2.activity_scores.index(min(self.l2.activity_scores))

            self.l2.add_to_slot(min_score_idx, {"id": lid, "content": line}, iteration)
            self.promoted.add(lid)
            self.last_injected_line = line  # Śledzimy tylko ostatnią, jeśli jest więcej
            self.last_injected_slot_idx = min_score_idx
            injected_count += 1  # Zwiększamy licznik
            app.logger.info(
                f"Wstrzyknięto z pliku L2 '{line}' do slotu {min_score_idx} (podobieństwo: {current_sim:.2f}). Całkowicie wstrzyknięto: {injected_count}.")

            # NIE DODAJEMY `break` tutaj, kontynuujemy pętlę, aby wstrzyknąć więcej, jeśli dostępne

        # Ten blok sprawdza, czy nic nie zostało wstrzyknięte po przejściu przez wszystkie linie
        if injected_count == 0:
            app.logger.info("Brak relewantnych i nieodfiltrowanych linii z pliku do wstrzyknięcia do L2.")

    def retrieve_context(self, current_input, memory_top_k=2, l2_sim_threshold=0.01):
        context_pieces = []

        if hasattr(self, 'l1'):
            context_pieces.extend(self.l1.get_all_contents())
        else:
            app.logger.warning("L1 (Short-term memory) nie jest zainicjalizowane w ACRMechanism.")

        active_l2_contents = sorted(
            [(self.l2.slots[i]['content'], self.l2.activity_scores[i])
             for i in range(len(self.l2.slots)) if self.l2.slots[i]['content']],
            key=lambda item: item[1], reverse=True
        )

        for entry_content, score in active_l2_contents[:memory_top_k]:
            if score > l2_sim_threshold:
                context_pieces.append(f"Fakt z pamięci L2: {entry_content}")

        return "\n".join(list(dict.fromkeys(context_pieces)))


# === Init models & memories ===

app = Flask(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

L3_FILE = "l3_memory.pkl"
L2_FILE = "plik.txt"
L2_CACHE = "plik_embeddings_cache.pkl"

app.logger.info("Rozpoczynam ładowanie modeli i pamięci...")

encoder = SentenceTransformer('all-MiniLM-L6-v2')

# --- TUTAJ BĘDZIE ZMIANA DLA  API ---
# Usuń linie dotyczące ładowania lokalnego modelu Hugging Face:
# polish_llm_name = "microsoft/Phi-3-mini-4k-instruct"
# tokenizer = AutoTokenizer.from_pretrained(polish_llm_name)
# lm_model = AutoModelForCausalLM.from_pretrained(...)
# lm_model.eval()
# if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id

# DODAJ KONFIGURACJĘ API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")# Pobierz klucz z zmiennej środowiskowej

if not OPENAI_API_KEY:
    app.logger.error("Błąd: Brak klucza API OpenAI. Ustaw zmienną środowiskową OPENAI_API_KEY.")
    raise ValueError("OPENAI_API_KEY environment variable not set.")

openai.api_key = OPENAI_API_KEY # Ustaw klucz dla biblioteki openai
app.logger.info("OpenAI API skonfigurowane pomyślnie!")
# --- KONIEC ZMIAN DLA GEMINI API ---


l1 = L1Memory()
l2 = L2Memory(slot_count=5)
l3 = L3Memory(encoder)
acr = ACRMechanism(l2, encoder, L2_FILE, L2_CACHE)

acr.l1 = l1  # Przypisujemy instancję L1 do ACR

l3.load_from_file(L3_FILE)

if not l1.get_all_contents():
    l1.add("AIram: Cześć! Jak mogę pomóc?")
    l1.add("Użytkownik: Pokaż mi przykład działania pamięci.")

if not l2.get_all_contents():
    l2.add_to_slot(0, {"id": "slot0", "content": "AIram to asystent AI stworzony do rozmów w języku polskim."}, 0)
    l2.add_to_slot(1, {"id": "slot1", "content": "Stolica Polski to Warszawa."}, 0)
    l2.add_to_slot(2, {"id": "slot2", "content": "Pies to ssak domowy."}, 0)

if not l3.entries:
    l3.add("AIram_info", "AIram jest asystentem AI z zaawansowaną architekturą pamięci.")
    l3.add("Projekt_info", "Projekt AIram ma na celu stworzenie inteligentnego interfejsu konwersacyjnego.")

# === Flask app HTML + JavaScript ===
# ... (HTML i JavaScript pozostają bez zmian - już je załączyłeś/aś) ...
HTML = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="utf-8">
    <title>AIram Chat</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }
        .container { max-width: 900px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1, h2 { color: #0056b3; }
        form { margin-bottom: 20px; display: flex; flex-wrap: wrap; align-items: flex-end;}
        .form-group { margin-right: 15px; margin-bottom: 10px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .form-group input[type="number"], .form-group input[type="range"] { width: 80px; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        .form-group input[type="range"] { width: 120px; vertical-align: middle; }
        input[type="text"] { flex-grow: 1; padding: 10px; margin-right: 10px; border: 1px solid #ddd; border-radius: 4px; }
        button { padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin-left: 5px; }
        button:hover { background-color: #0056b3; }
        #chat_history { border: 1px solid #eee; padding: 15px; border-radius: 4px; background-color: #fdfdfd; max-height: 400px; overflow-y: auto; margin-bottom: 20px; }
        .message { margin-bottom: 10px; }
        .user-msg { text-align: right; color: #0056b3; }
        .ai-msg { text-align: left; color: #28a745; }
        .memory-section { margin-top: 30px; border-top: 1px solid #eee; padding-top: 20px; }
        .slider-val { display: inline-block; width: 30px; text-align: center; }
        /* --- NOWE STYLE DLA PODGLĄDU PAMIĘCI --- */
        .memory-detail { background-color: #e9e9e9; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
        .memory-detail h3 { margin-top: 0; color: #0056b3; }
        .memory-slot { border: 1px solid #ccc; padding: 5px; margin-bottom: 5px; background-color: #f8f8f8; border-radius: 3px; font-size: 0.9em; }
        .low-score { opacity: 0.6; }
        /* --- KONIEC NOWYCH STYLÓW --- */
    </style>
</head>
<body>
    <div class="container">
        <h1>AIram - Twój Asystent AI</h1>
        <div id="chat_history"></div>

        <form id="chat_form">
            <div class="form-group">
                <label for="max_new_tokens">Max Tokens:</label>
                <input type="number" id="max_new_tokens" value="20" min="10" max="512">
            </div>
            <div class="form-group">
                <label for="temperature">Temp (<span id="temp_val">0.7</span>):</label>
                <input type="range" id="temperature" min="0.1" max="2.0" step="0.1" value="0.7">
            </div>
            <div class="form-group">
                <label for="top_k_llm">Top K (LLM):</label>
                <input type="number" id="top_k_llm" value="50" min="0" max="200">
            </div>
            <div class="form-group">
                <label for="top_p_llm">Top P (LLM) (<span id="top_p_val">0.95</span>):</label>
                <input type="range" id="top_p_llm" min="0.0" max="1.0" step="0.05" value="0.95">
            </div>
            <div class="form-group">
                <label for="memory_top_k">Pamięć Top K:</label>
                <input type="number" id="memory_top_k" value="2" min="1" max="10">
            </div>
            <div class="form-group">
                <label for="l2_sim_threshold">Próg Pod. L2 (<span id="l2_sim_val">0.6</span>):</label>
                <input type="range" id="l2_sim_threshold" min="0.0" max="0.6" step="0.01" value="0.59">
            </div>
            <div class="form-group">
                <label for="l2_decay_rate">Spadek L2 (<span id="l2_decay_val">0.95</span>):</label>
                <input type="range" id="l2_decay_rate" min="0.0" max="1.0" step="0.01" value="0.38">
            </div>

            <input type="text" id="user_input" name="user_input" placeholder="Wpisz wiadomość..." autocomplete="off" autofocus>
            <button type="submit">Wyślij</button>
            <button type="button" onclick="showMemory()">Odśwież pamięć</button> </form>

        <div class="memory-section">
            <h2>Podgląd Pamięci i ACR (Real-time):</h2>
            <div id="memory_display">
                <div class="memory-detail">
                    <h3>Pamięć L1 (Historia rozmowy)</h3>
                    <pre id="l1_contents" style="white-space:pre-wrap;"></pre>
                </div>
                <div class="memory-detail">
                    <h3>Pamięć L2 (Kontekstowa)</h3>
                    <p>Ostatnio **wstrzyknięto** do L2 (z pliku): <span id="last_injected_l2">Brak</span> (Slot: <span id="last_injected_slot_idx">Brak</span>)</p>
                    <p>Ostatnia **aktywacja** w L2 (przez zapytanie): <span id="last_queried_l2">Brak</span> (Podobieństwo: <span id="last_queried_sim">Brak</span>)</p>
                    <div id="l2_slots_display"></div>
                </div>
                <div class="memory-detail">
                    <h3>Pamięć L3 (Długoterminowa)</h3>
                    <pre id="l3_contents" style="white-space:pre-wrap;"></pre>
                </div>
                <div class="memory-detail">
                    <h3>Zawartość pliku 'plik.txt'</h3>
                    <pre id="file_txt_contents" style="white-space:pre-wrap; max-height: 200px; overflow-y: auto;"></pre>
                </div>
            </div>
        </div>
        </div>

    <script>
        const chatForm = document.getElementById('chat_form');
        const userInput = document.getElementById('user_input');
        const chatHistory = document.getElementById('chat_history');
        // const memoryDiv = document.getElementById('memory'); // Ta zmienna już nie jest potrzebna dla ogólnego bloku

        const fileTxtContentsDiv = document.getElementById('file_txt_contents');
        const tempSlider = document.getElementById('temperature');
        const tempValSpan = document.getElementById('temp_val');
        const topPSlider = document.getElementById('top_p_llm');
        const topPValSpan = document.getElementById('top_p_val');
        const l2SimSlider = document.getElementById('l2_sim_threshold');
        const l2SimValSpan = document.getElementById('l2_sim_val');
        const l2DecaySlider = document.getElementById('l2_decay_rate');
        const l2DecayValSpan = document.getElementById('l2_decay_val');

        // --- NOWE: Elementy DOM dla podglądu pamięci ---
        const l1ContentsDiv = document.getElementById('l1_contents');
        const l2SlotsDisplayDiv = document.getElementById('l2_slots_display');
        const l3ContentsDiv = document.getElementById('l3_contents');
        const lastInjectedL2Span = document.getElementById('last_injected_l2');
        const lastInjectedSlotIdxSpan = document.getElementById('last_injected_slot_idx');
        const lastQueriedL2Span = document.getElementById('last_queried_l2');
        const lastQueriedSimSpan = document.getElementById('last_queried_sim');
        // --- KONIEC NOWE ---

        tempSlider.oninput = () => tempValSpan.textContent = tempSlider.value;
        topPSlider.oninput = () => topPValSpan.textContent = topPSlider.value;
        l2SimSlider.oninput = () => l2SimValSpan.textContent = l2SimSlider.value;
        l2DecaySlider.oninput = () => l2DecayValSpan.textContent = parseFloat(l2DecaySlider.value).toFixed(2);

        tempValSpan.textContent = tempSlider.value;
        topPValSpan.textContent = topPSlider.value;
        l2SimValSpan.textContent = l2SimSlider.value;
        l2DecayValSpan.textContent = parseFloat(l2DecaySlider.value).toFixed(2);


        chatForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const userMessage = userInput.value;
            if (!userMessage.trim()) return;

            chatHistory.innerHTML += `<p class="message user-msg"><b>Ty:</b> ${userMessage}</p>`;
            userInput.value = '';

            const params = {
                message: userMessage,
                max_new_tokens: parseInt(document.getElementById('max_new_tokens').value),
                temperature: parseFloat(document.getElementById('temperature').value),
                top_k_llm: parseInt(document.getElementById('top_k_llm').value),
                top_p_llm: parseFloat(document.getElementById('top_p_llm').value),
                memory_top_k: parseInt(document.getElementById('memory_top_k').value),
                l2_sim_threshold: parseFloat(document.getElementById('l2_sim_threshold').value),
                l2_decay_rate: parseFloat(document.getElementById('l2_decay_rate').value)
            };

            try {
                const r = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(params)
                });
                const j = await r.json();
                chatHistory.innerHTML += `<p class="message ai-msg"><b>AIram:</b> ${j.reply}</p>`;
                chatHistory.scrollTop = chatHistory.scrollHeight;
                showMemory(); // --- NOWE: Odśwież pamięć po każdej konwersacji ---
            } catch (error) {
                console.error('Błąd:', error);
                chatHistory.innerHTML += `<p class="message ai-msg" style="color: red;"><b>AIram:</b> Wystąpił błąd podczas komunikacji.</p>`;
            }
        });
        async function showFileContent(){
        try {
            let r = await fetch("/file_content");
            let j = await r.json();
            fileTxtContentsDiv.textContent = j.content;
        } catch (error) {
            console.error('Błąd pobierania pliku txt:', error);
            fileTxtContentsDiv.textContent = "Błąd ładowania zawartości pliku 'plik.txt'.";
        }
    }




        async function showMemory(){
            // memoryDiv.textContent = "Ładowanie pamięci..."; // Już nie potrzebne dla ogólnego bloku
            try {
                let r = await fetch("/memory");
                let j = await r.json();

                // --- NOWE: Wyświetlanie L1 ---
                l1ContentsDiv.textContent = j.L1_current_conversation.join('\\n');

                // --- NOWE: Wyświetlanie L2 ---
                lastInjectedL2Span.textContent = j.ACR_last_injected_line || 'Brak';
                lastInjectedSlotIdxSpan.textContent = j.ACR_last_injected_slot_idx !== null ? j.ACR_last_injected_slot_idx : 'Brak';
                lastQueriedL2Span.textContent = j.ACR_last_queried_line || 'Brak';
                lastQueriedSimSpan.textContent = j.ACR_last_queried_sim !== null ? j.ACR_last_queried_sim.toFixed(2) : 'Brak';

                l2SlotsDisplayDiv.innerHTML = ''; // Wyczyść stare sloty
                j.L2_slots_and_activity.forEach(slot => {
                    const slotDiv = document.createElement('div');
                    // Dodaj klasę CSS dla slotów o niskiej aktywności
                    const scoreClass = slot.score < 0.2 ? 'low-score' : ''; // Próg można dostosować
                    slotDiv.className = `memory-slot ${scoreClass}`;
                    slotDiv.innerHTML = `<b>Aktywność:</b> ${slot.score.toFixed(4)}<br><b>Treść:</b> ${slot.content}`;
                    l2SlotsDisplayDiv.appendChild(slotDiv);
                });

                // --- NOWE: Wyświetlanie L3 ---
                l3ContentsDiv.textContent = JSON.stringify(j.L3_entries, null, 2);

            } catch (error) {
                console.error('Błąd pobierania pamięci:', error);
                l1ContentsDiv.textContent = "Błąd ładowania pamięci L1.";
                l2SlotsDisplayDiv.innerHTML = "Błąd ładowania pamięci L2.";
                l3ContentsDiv.textContent = "Błąd ładowania pamięci L3.";
            }
        }

        // --- NOWE: Automatyczne odświeżanie pamięci co kilka sekund ---
        setInterval(showMemory, 5000); // Odświeżaj co 5 sekund (5000 ms)
        showMemory(); // Wywołaj raz na początku, aby załadować początkowy stan pamięci
        // --- KONIEC NOWE ---
         setInterval(showMemory, 5000); // Odświeżaj co 5 sekund
        showMemory(); // Wywołaj raz na początku

        // --- NOWE: Automatyczne odświeżanie pliku txt co kilka sekund ---
        setInterval(showFileContent, 10000); // Odświeżaj plik co 10 sekund (10000 ms)
        showFileContent(); // Wywołaj raz na
    </script>
</body>
</html>
"""


# --- 5. Funkcja generująca odpowiedź (gen_resp) ---

def gen_resp(user_input, max_new_tokens, temperature, top_k_llm, top_p_llm, memory_top_k, l2_sim_threshold,
             l2_decay_rate):
    app.logger.info(f"--- Generowanie odpowiedzi dla zapytania: '{user_input}' ---")
    app.logger.info(
        f"Parametry LLM: max_tokens={max_new_tokens}, temp={temperature}, top_k={top_k_llm}, top_p={top_p_llm}")
    app.logger.info(
        f"Parametry pamięci: mem_top_k={memory_top_k}, l2_sim_thresh={l2_sim_threshold}, l2_decay_rate={l2_decay_rate}")

    retrieved_context = acr.retrieve_context(user_input, memory_top_k=memory_top_k, l2_sim_threshold=l2_sim_threshold)
    num_recent_l1_messages = 4  # Np. 2 wymiany (Użytkownik, AIram) * 2
    l1_context_list = l1.get_all_contents()[-num_recent_l1_messages:]  # Pobierz tylko N ostatnich
    l1_context = "\n".join(l1_context_list)

    # --- ZMIENIONY SPOSÓB TWORZENIA PROMPTU I WYWOŁANIA DLA OPENAI API ---
    messages_for_llm = [
        {"role": "system",
         "content": "Jesteś AIramem, asystentem AI. Odpowiadaj zwięźle i precyzyjnie. Koniecznie wykorzystuj dostarczony kontekst i historię rozmowy, jeśli są relewantne. Jeśli nie masz wystarczających informacji, powiedz 'Nie wiem.'. Nie dodawaj zbędnych komentarzy ani emotikon. Unikaj powtórzeń."},
    ]

    if retrieved_context:
        messages_for_llm.append({"role": "system", "content": f"Kontekst z pamięci długoterminowej: {retrieved_context}"})
    if l1_context:
        # Dodajemy historię rozmowy do promptu w roli 'user' jako dodatkowy kontekst.
        # W GPT API nie ma bezpośredniego mapowania "AIram: " i "Użytkownik: " na role 'model'/'user' w historii systemowej,
        # więc traktujemy to jako ciągły kontekst od systemu.
        messages_for_llm.append({"role": "system", "content": f"Historia rozmowy: {l1_context}"})

    messages_for_llm.append({"role": "user", "content": user_input})

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", # Możesz zmienić na "gpt-4o", "gpt-4-turbo" itp., jeśli masz dostęp i chcesz lepszą jakość
            messages=messages_for_llm,
            max_tokens=max_new_tokens,
            temperature=temperature,
            # top_k nie jest bezpośrednio używane w tym wywołaniu API OpenAI dla chat completions.
            top_p=top_p_llm, # top_p jest używane
        )
        reply = response.choices[0].message.content.strip()
        app.logger.info(f"Odpowiedź LLM: {reply}")

    except Exception as e:
        app.logger.error(f"Błąd podczas komunikacji z API OpenAI: {e}")
        reply = "Przepraszam, wystąpił błąd podczas generowania odpowiedzi."

    # --- Aktualizacja L1 (pamięć konwersacji) ---
    l1.add(f"Użytkownik: {user_input}")
    l1.add(f"AIram: {reply}")

    # --- Konsolidacja L2 ---
    l2.decay_and_consolidate_l2(decay_rate=l2_decay_rate)
    # --- Aktywacja L2 na podstawie zapytania użytkownika ---
    query_embedding = encoder.encode(user_input, convert_to_tensor=True).cpu()
    l2.query_l3_and_activate_l2(query_embedding, encoder, memory_top_k=memory_top_k)
    # --- Wstrzyknięcie relewantnych faktów z pliku do L2 ---
    acr.inject_relevant_to_l2(iteration=0, query_emb=query_embedding,
                              l2_sim_threshold=l2_sim_threshold)

    app.logger.info(f"AIram Reply: {reply}")
    return reply


# === Flask Routes ===
# ... (pozostałe route'y /memory i /file_content bez zmian) ...
app = Flask(__name__)


# ... (konfiguracja loggera i inicjalizacja obiektów pamięci) ...

@app.route('/')
def index():
    return HTML


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    max_new_tokens = data.get('max_new_tokens', 20)
    temperature = data.get('temperature', 0.7)
    top_k_llm = data.get('top_k_llm', 50)
    top_p_llm = data.get('top_p_llm', 0.95)
    memory_top_k = data.get('memory_top_k', 2)
    l2_sim_threshold = data.get('l2_sim_threshold', 0.6)
    l2_decay_rate = data.get('l2_decay_rate', 0.95)

    reply = gen_resp(user_input, max_new_tokens, temperature, top_k_llm, top_p_llm, memory_top_k, l2_sim_threshold,
                     l2_decay_rate)
    return jsonify({"reply": reply})


@app.route('/memory')
def get_memory_status():
    l1_contents = l1.get_all_contents()
    l2_slots_and_activity = [
        {"id": slot['id'], "content": slot['content'], "score": l2.activity_scores[i]}
        for i, slot in enumerate(l2.slots) if slot['content']
    ]
    l3_entries_display = {k: v for k, v in l3.entries.items()}  # Możesz dostosować, co chcesz pokazać z L3

    return jsonify({
        "L1_current_conversation": l1_contents,
        "L2_slots_and_activity": l2_slots_and_activity,
        "L3_entries": l3_entries_display,
        "ACR_last_injected_line": acr.last_injected_line,
        "ACR_last_injected_slot_idx": acr.last_injected_slot_idx,
        "ACR_last_queried_line": l2.last_activated_content,  # Z L2
        "ACR_last_queried_sim": l2.last_activated_sim  # Z L2
    })


@app.route('/file_content')
def get_file_content():
    try:
        with open(L2_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        return jsonify({"content": content})
    except FileNotFoundError:
        return jsonify({"content": "Plik 'plik.txt' nie istnieje."}), 404
    except Exception as e:
        app.logger.error(f"Błąd podczas czytania pliku L2: {e}")
        return jsonify({"content": f"Błąd podczas czytania pliku L2: {e}"}), 500


# --- Uruchomienie aplikacji ---
if __name__ == '__main__':
    # Usunięcie starych plików cache przy starcie, aby zapewnić świeży start (opcjonalnie)
    if os.path.exists(L2_CACHE):
        os.remove(L2_CACHE)
        app.logger.info(f"Usunięto stary cache L2: {L2_CACHE}")

    app.run(debug=True, port=5000)
