# AIram  
**Innowacja to nie technologia, to sposób myślenia**

---

## Filozofia projektu  
AIram to nie tylko technologia – to nowy sposób myślenia o pamięci w sztucznej inteligencji.  
Zamiast polegać wyłącznie na monolitycznych modelach i gigantycznych bazach danych, AIram proponuje trójwarstwową architekturę pamięci, która łączy trwałość, elastyczność i efektywność.  

Ten model inspiruje się naturalnymi procesami zapamiętywania i przypominania, tworząc system odporny na zapominanie i zdolny do adaptacyjnego zarządzania wiedzą.

## Wprowadzenie  
AIram to lekka, efektywna i trwała architektura pamięci AI, łącząca trzy warstwy pamięci:  
- **L1** – krótkoterminowa, dynamiczna pamięć konwersacji  
- **L2** – operacyjna pamięć o ograniczonej liczbie slotów z aktywnością  
- **L3** – trwała, nieulotna pamięć zapisywana na dysku  

Mechanizm **ACR (Aktywny Mechanizm Odzyskiwania Kontekstu)** inteligentnie promuje wiedzę z trwałego archiwum tekstowego do operacyjnej pamięci L2.

## Architektura  
- **L1** przechowuje bieżące konwersacje i szybkie fakty.  
- **L2** zarządza slotami pamięci z aktywnością, która zanika i wzmacnia się w zależności od zapytań.  
- **L3** to wolniej aktualizowana baza trwałej wiedzy.

Mechanizm ACR dopasowuje kontekst z archiwum plikowego do bieżących potrzeb, promując istotne informacje do L2 i zapewniając trwałość.

## Implementacja  
- Embeddingi tekstów generowane są przez **SentenceTransformers**.  
- Odpowiedzi generuje **GPT-2**.  
- Interfejs webowy zbudowany na **Flask**.  
- Dane L3 zapisywane w formacie pickle, L2 w pliku tekstowym z cache embeddingów.

## Zalety  
- Odporność na zapominanie dzięki trójwarstwowej pamięci i archiwum.  
- Efektywne, adaptacyjne zarządzanie pamięcią krótkoterminową i operacyjną.  
- Lekkość i możliwość działania offline na ograniczonych zasobach.  
- Modularność, łatwość rozbudowy i integracji.

## Perspektywy  
- Integracja z większymi, nowoczesnymi modelami językowymi.  
- Rozwój mechanizmu ACR o inteligentniejszą selekcję i aktualizację.  
- Tworzenie benchmarków do porównań i oceny jakości pamięci.  
- Zastosowania w asystentach, robotyce, IoT i edukacji.

---


**Autor:** Tomasz Kiliańczyk  
**Rok:** 2025  
**Projekt – AIram**

