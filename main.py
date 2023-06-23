import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Kod źródłowy projektu dotyczącego dostępu do antykoncepcji oraz wpływu różnych czynników na wskaźnik dzietności.
# Autor: Weronika Łoś


# wczytanie danych
data_contraception = pd.read_csv("contraceptive-prevalence-any-methods-of-women-ages-15-49.csv")
data_education = pd.read_csv("fertility-rate-vs-mean-years-of-schooling (1).csv")

if os.path.exists('merged.csv'):
    pass
else:
    merged_data_to_save = pd.merge(data_contraception, data_education, on=["Entity", "Year"])
    merged_data_to_save.to_csv('merged.csv')
merged_data = pd.read_csv("merged.csv")


# zmiany wskaźnika dzietności w wybranych krajach na przestrzeni lat
def plot_fertility_trends(data, countries):
    plt.figure(figsize=(10, 6))
    for country in countries:
        data = data[(data['Year'] >= 1950) & (data['Year'] <= 2023)]
        country_data = data[data["Entity"] == country]
        plt.plot(country_data["Year"], country_data["Fertility rate"], label=country)
    plt.xlabel("Rok")
    plt.ylabel("Wskaźnik dzietności")
    plt.title("Zmiany wskaźnika dzietności")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.show()


# związek między stosowaniem antykoncepcji a wskaźnikiem dzietności oraz współczynnik korelacji
def fertility_contraception_correlation(data):
    # wyodrębnienie potrzebnych kolumn
    df = data[['Contraceptive prevalence, any method (% of married women ages 15-49)', 'Fertility rate']]

    # usunięcie pustych wierszy
    df = df.dropna()

    # konwersja na odpowiednie typy danych
    df['Contraceptive prevalence, any method (% of married women ages 15-49)'] = df[
        'Contraceptive prevalence, any method (% of married women ages 15-49)'].astype(float)
    df['Fertility rate'] = df['Fertility rate'].astype(float)

    # tworzenie wykresu
    sns.regplot(x='Contraceptive prevalence, any method (% of married women ages 15-49)', y='Fertility rate', data=df,
                line_kws={'color': 'red'})
    plt.title('Dostęp do metod antykoncepcji i współczynnik dzietności')
    plt.xlabel('Kobiety stosujące metody antykoncepcji [ % ]')
    plt.ylabel('Współczynnik dzietności')
    plt.show()

    # współczynnik korelacji
    correlation = df['Contraceptive prevalence, any method (% of married women ages 15-49)'].corr(df['Fertility rate'])
    print('Współczynnik korelacji:', correlation)


def fertility_education_correlation(data):
    # wyodrębnienie potrzebnych kolumn
    df = data[['Total years of schooling (Lee-Lee (2016))', 'Fertility rate']]

    # usunięcie pustych wierszy
    df = df.dropna()

    # konwersja na odpowiednie typy danych
    df['Total years of schooling (Lee-Lee (2016))'] = df['Total years of schooling (Lee-Lee (2016))'].astype(float)
    df['Fertility rate'] = df['Fertility rate'].astype(float)

    # tworzenie wykresu
    sns.regplot(x='Total years of schooling (Lee-Lee (2016))', y='Fertility rate', data=df, line_kws={'color': 'red'})
    plt.title('Związek między dzietnością a latami nauki')
    plt.xlabel('Średnia długość edukacji(w latach)')
    plt.ylabel('Współczynnik dzietności')
    plt.show()

    # współczynnik korelacji
    correlation = df['Total years of schooling (Lee-Lee (2016))'].corr(df['Fertility rate'])
    print('Współczynnik korelacji:', correlation)


def classification_model(data):
    # wyodrębnienie potrzebnych kolumn
    df = data[['Contraceptive prevalence, any method (% of married women ages 15-49)',
               'Total years of schooling (Lee-Lee (2016))', 'Fertility rate']]

    # usunięcie pustych wierszy
    df = df.dropna()

    # konwersja na odpowiednie typy danych
    df['Contraceptive prevalence, any method (% of married women ages 15-49)'] = df[
        'Contraceptive prevalence, any method (% of married women ages 15-49)'].astype(float)
    df['Total years of schooling (Lee-Lee (2016))'] = df['Total years of schooling (Lee-Lee (2016))'].astype(float)
    df['Fertility rate'] = df['Fertility rate'].astype(float)

    # Podział danych na cechy X i etykiety y
    X = df[['Contraceptive prevalence, any method (% of married women ages 15-49)',
            'Total years of schooling (Lee-Lee (2016))']]
    y = df['Fertility rate']

    # Konwersja wskaźnika dzietności na "niski" oraz "wysoki"
    y = y.apply(lambda x: 'wysoki' if x > 3 else 'niski')

    # zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # dopasowanie modelu KNN
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    # predykcja dla zbioru testowego
    y_pred = model.predict(X_test)

    # obliczenie dokładności
    accuracy = accuracy_score(y_test, y_pred)
    print('Dokładność modelu:', accuracy)

    plt.scatter(X_train['Contraceptive prevalence, any method (% of married women ages 15-49)'],
                X_train['Total years of schooling (Lee-Lee (2016))'], c=y_train.map({'niski': 'blue', 'wysoki': 'red'}))
    plt.xlabel('Dostęp do antykoncepcji')
    plt.ylabel('Lata nauki')
    plt.title('Dane treningowe')
    plt.show()

    plt.scatter(X_test['Contraceptive prevalence, any method (% of married women ages 15-49)'],
                X_test['Total years of schooling (Lee-Lee (2016))'], c=y_test.map({'niski': 'blue', 'wysoki': 'red'}))
    plt.xlabel('Dostęp do antykoncepcji')
    plt.ylabel('Lata nauki')
    plt.title('Dane testowe')
    plt.show()


# dla roku 2015
def data_overview_continents(data):
    # grupowanie danych po kontynentach i obliczanie średnich wartości
    continent_data = data.groupby('Continent').mean('Fertility rate')
    continent_data_2015 = continent_data[(continent_data['Year'] == 2015)]
    print(continent_data_2015)
    if os.path.exists('continent_data.csv'):
        pass
    else:
        continent_data.to_csv('continent_data.csv')

    # wykres średniego wskaźnika dzietności dla poszczególnych kontynentów
    plt.bar(continent_data_2015.index, continent_data_2015['Fertility rate'], width=0.5)
    plt.ylabel('Średni wskaźnik dzietności')
    plt.title('Średni wskaźnik dzietności dla poszczególnych kontynentów')
    plt.figure(figsize=(3, 4))
    plt.show()

    # wykres średniego wskaźnika dostępności antykoncepcji dla poszczególnych kontynentów
    plt.bar(continent_data.index,
            continent_data['Contraceptive prevalence, any method (% of married women ages 15-49)'], width=0.5)
    plt.ylabel('Średni wskaźnik dostępności antykoncepcji')
    plt.title('Dostępność antykoncepcji dla poszczególnych kontynentów')
    plt.show()


if __name__ == "__main__":
    plot_fertility_trends(data_education, ['Poland', 'Afghanistan', 'Colombia', 'Nigeria', 'United States'])
    fertility_contraception_correlation(merged_data)
    fertility_education_correlation(merged_data)
    classification_model(merged_data)
    data_overview_continents(merged_data)
