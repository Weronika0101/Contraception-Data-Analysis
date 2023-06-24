# MSID_project

W ramach kursu Metody Systemowe i Decyzyjne podjęłam się projektu dotyczącego dostępu do antykoncepcji oraz wpływu różnych czynników na wskaźnik
dzietności. Celem mojej pracy jest dokładne zbadanie korelacji między czynnikami, takimi jak dostęp do edukacji i antykoncepcji, a współczynnikiem dzietności
w wybranych krajach na przestrzeni lat . Poprzez przeprowadzenie analizy, dążę
do wyciągnięcia trafnych wniosków na temat tych zależności.

Aby dokonać analizy podanych zbiorów danych, znalezienia korelacji między
nimi oraz ich wizualizacji, zastosowałam następujące metody:
- Przedstawienie i porównanie danych na wykresie
- Obliczenie współczynników korelacji
- Stworzenie modelu klasyfikacji i obliczenie jego dokładności

Wybrałam powyższe metody, ponieważ umożliwią mi one badanie zależności między danymi zawartymi w wybranych zbiorach. Przedstawienie danych na
wykresie pozwoli mi porównać zmiany wartości wskaźnika dzietności w różnych
krajach na przestrzeni lat, co dostarczy informacji na temat szybkości spadku
tego wskaźnika oraz np. momentu, w którym zaczynał maleć w poszczególnych
krajach. Grupowanie danych i tworzenie wykresów pozwoli mi również zidentyfikować różnice w wartości średniego wskaźnika dzietności oraz dostępu do
antykoncepcji w danym roku na poszczególnych kontynentach.
Na podstawie obliczonego współczynnika korelacji będę w stanie określić,
czy istnieje silna korelacja między dostępem do antykoncepcji a wskaźnikiem
dzietności, czy jest ona nieznacząca. Podobną zależność zbadam dla związku
pomiędzy średnią długością edukacji szkolnej w latach a wskaźnikiem dzietności.
Do powyższych analiz skorzystam z bibliotek pandas, matplotlib oraz seaborn.
Zastosowanie modelu klasyfikacji pozwoli na przewidzenie, czy dany kraj
ma wysoki czy niski wskaźnik dzietności na podstawie dostępnych danych dotyczących antykoncepcji i edukacji, tutaj użyję KNeighborsClassifier z biblioteki
sklearn, który implementuje algorytm k- najbliższych sąsiadów.


Szczegółowy raport z analizy wraz z wnioskami znajduje się w pliku raport.pdf
