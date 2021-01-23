# INL Zadanie zaliczeniowe

## Cześć statystyczna
Metoda statystyczna uczenia maszynowego rozwiązana za pomocą
NKJPStatisticalApproach2 w pliku `main.py`.

Rozwiązanie opiera się na załadowaniu danych:
* `nkjp-morph-named.txt` - w celu dostępu do tagów
* `nkjp-tab-onlypos.txt' - w celu podziału na prawidłowe zdania
* modelu nlp z `pl_core_news_lg` - w celu analizy kontekstu zdań

Rozwiązanie podczas nauki bierze pod uwagę:
* standardowe parametry słowa (np. czy jest liczbą, czy jest wielkimi literami)
* kontekst w zdaniu - zależność do ojca i jego lemat
* pozycja w zdaniu - (koniec/początek)
* tagi morfologiczne swoje oraz słów poprzedzających i kolejnych

Wyniki uzyskane:
* Precyzja: 0.864
* Pełność: 0.785
* Miara F1: 0.820

## Część sieci neuronowych
Do rozwiązania tej części postanowiłem zabrać się za problem 1 tegorocznego
konkursu POLEVAL.

Zadanie polega na zamianie słów wygenerowanych przez analizator mowy na ich
prawidłowe odpowiedniki.
Danymi wejściowymi są słowa wygenerowane przez analizator oraz dla porównania
transkrypcje słów, które były mówione.

Podstawowym wyzwaniem było przygotowanie danych, bo nie posiadamy jednoznacznego
odwzorowania które słowo konkretnie odpowiada któremu.

Napisałem algorytm z grubsza działający na zasadzie iteracji po słowach
docelowych i znajdywaniu ich najbardziej podobnych odpowiedników na zbliżonych
miejscach w wypowiedzi. Użyta została funkcja `similar`, działająca w zbliżony
sposób do odległości Levensteina.

Jako, że sieć neuronowa z całych słów stanowiła zbyt duży zbiór na moje
możliwości techniczne po przeporządkowaniu słów zostały one podzielone na
coś w rodzaju sylab - każda część kończy się samogłoską.

Przy użyciu takiego rozwiązania moja sieć neuronowa uzyskuje precyzję rzędu
0.999.

Model bardzo dobrze uczy się danych wejściowych, a na danych testowych
nieznacznie rośnie.

Fakt, że dobrze uczy się danych wejściowych nie jest w tym przepadku zły,
bo zgodnie z opisem zadania wyłapanie typowych błędów dla konkretnego
analizatora mowy jest również przydatne.