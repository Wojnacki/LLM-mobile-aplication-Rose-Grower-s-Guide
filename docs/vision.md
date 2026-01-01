# Vision – Rose Grower’s Guide (LLM Offline)

## 1. Cel aplikacji
Celem aplikacji jest dostarczenie **mobilnego, działającego offline asystenta AI**,
który odpowiada na pytania dotyczące **uprawy i pielęgnacji róż**, w oparciu o:
- lokalny model językowy (LLM)
- sprawdzoną wiedzę ogrodniczą
- mechanizm Retrieval-Augmented Generation (RAG)

Aplikacja jest przeznaczona dla:
- amatorów ogrodnictwa
- hobbystów uprawiających róże
- użytkowników bez stałego dostępu do Internetu

---

## 2. Zakres MVP (Must-have)

### 2.1 Funkcjonalności
MVP aplikacji musi umożliwiać:

- zadawanie pytań w języku naturalnym (PL)
- uzyskiwanie odpowiedzi **offline**
- odpowiedzi oparte na:
  - lokalnej bazie wiedzy
  - kontekście pobranym przez RAG
- uruchomienie na:
  - Android
  - iOS

---

### 2.2 Zakres wiedzy (MVP)
W pierwszej wersji aplikacja obejmuje następujące obszary:

1. **Cięcie róż**
   - wiosenne
   - jesienne
   - sanitarne

2. **Choroby i szkodniki**
   - czarna plamistość
   - mączniak
   - mszyce

3. **Nawożenie**
   - kiedy nawozić
   - czym nawozić
   - objawy niedoborów

4. **Zimowanie róż**
   - kopczykowanie
   - okrywanie
   - różnice między odmianami

---

## 3. Poza zakresem MVP (Not now)
Celowo **nie realizujemy** w MVP:

- rozpoznawania obrazu (zdjęcia liści)
- synchronizacji z chmurą
- kont użytkowników
- personalizacji pod konkretne odmiany
- integracji z API pogodowym

Te elementy są **kandydatami na wersje późniejsze**.

---

## 4. Wymagania techniczne

### 4.1 LLM
- model offline w formacie **GGUF**
- uruchamiany przez `llama.cpp` lub kompatybilny runtime
- zoptymalizowany pod urządzenia mobilne

---

### 4.2 RAG
- embeddingi generowane offline (desktop)
- indeks wektorowy (FAISS)
- na telefon trafia:
  - gotowy indeks
  - pocięta i oczyszczona wiedza

---

### 4.3 Aplikacja mobilna
- Flutter (jeden codebase)
- komunikacja z warstwą inference lokalnie
- brak zależności sieciowych

---

## 5. Kryteria sukcesu MVP

MVP uznajemy za zakończone, gdy:

- aplikacja działa w trybie offline
- użytkownik może zadać pytanie tekstowe
- odpowiedź:
  - jest po polsku
  - odnosi się do wiedzy o różach
  - nie halucynuje oczywistych błędów
- aplikacja uruchamia się na:
  - przynajmniej jednym Androidzie
  - symulatorze iOS

---

## 6. Następne kroki (po MVP)
- rozszerzenie bazy wiedzy
- lepsze promptowanie
- wersjonowanie wiedzy
- wsparcie dla innych roślin
