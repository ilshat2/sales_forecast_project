import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb


def load_data():
    """Загрузка и объединение данных из CSV-файлов"""
    df1 = pd.read_csv(
        '../data/raw/id_597.csv',
        parse_dates=['OpenDate.Typed', 'CloseTime'],
    )
    df2 = pd.read_csv(
        '../data/raw/id_16.csv',
        parse_dates=['OpenDate.Typed', 'CloseTime'],
    )
    df3 = pd.read_csv(
        '../data/raw/id_34.csv',
        parse_dates=['OpenDate.Typed', 'CloseTime'],
    )
    return pd.concat([df1, df2, df3], ignore_index=True)


def preprocess_data(df):
    # Удаление дубликатов
    df = df.drop_duplicates()
    # Заполнение пропусков
    df['DishDiscountSumInt'] = df['DishDiscountSumInt'].fillna(0)
    return df


def add_features(df):
    # Временные признаки
    df['DayOfWeek'] = df['OpenDate.Typed'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
    return df


def train_model(X, y):
    # Модель
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    return model


def make_predictions(model, tomorrow_date):
    # Логика прогнозирования
    pass


if __name__ == "__main__":
    print("Программа запущена")
