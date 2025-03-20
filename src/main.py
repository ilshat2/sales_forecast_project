import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb


def load_data():
    """Загрузка и объединение данных из файлов"""
    df1 = pd.read_csv(
        '../data/raw/id_597.xlsx',
        parse_dates=['OpenDate.Typed', 'CloseTime'],
    )
    df2 = pd.read_csv(
        '../data/raw/id_16.xlsx',
        parse_dates=['OpenDate.Typed', 'CloseTime'],
    )
    df3 = pd.read_csv(
        '../data/raw/id_34.xlsx',
        parse_dates=['OpenDate.Typed', 'CloseTime'],
    )
    return pd.concat([df1, df2, df3], ignore_index=True)


def preprocess_data(df):
    """Предобработка данных"""

    # Удаление дубликатов
    df = df.drop_duplicates()

    # Заполнение пропусков
    df['DishDiscountSumInt'] = df['DishDiscountSumInt'].fillna(0)

    # Извлечение времени закрытия чека
    df['CloseHour'] = df['CloseTime'].dt.hour
    df['CloseMinute'] = df['CloseTime'].dt.minute

    return df


def add_features(df):
    """Генерация новых признаков"""

    # Временные признаки
    df['DayOfWeek'] = df['OpenDate.Typed'].dt.dayofweek
    df['Month'] = df['OpenDate.Typed'].dt.month
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    # Сортировка для лаговых признаков
    df = df.sort_values(['DishName', 'RestorauntGroup', 'OpenDate.Typed'])

    # Лаги продаж за 3 дня
    df['Lag3'] = df.groupby(['DishName', 'RestorauntGroup'])['DishAmountInt'].shift(3)

    # Кодирование категорий
    le_dish = LabelEncoder()
    le_rest = LabelEncoder()
    df['DishEncoded'] = le_dish.fit_transform(df['DishName'])
    df['RestEncoded'] = le_rest.fit_transform(df['RestorauntGroup'])

    return df, le_dish, le_rest


def prepare_training_data(df):
    """Подготовка данных для обучения модели"""

    features = [
        'DishEncoded',
        'RestEncoded',
        'DayOfWeek',
        'Month',
        'CloseHour',
        'IsWeekend',
        'Lag3',
        'DishDiscountSumInt',
    ]
    target = 'DishAmountInt'

    # Удаление строк с пропусками
    df_clean = df.dropna(subset=features + [target])

    return df_clean[features], df_clean[target]


def train_model(X, y):
    """Обучение модели LightGBM"""

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Создание и обучение модели
    model = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100
    )
    model.fit(X_train, y_train)

    # Оценка модели
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print("Качество модели:")
    print(f"Обучающая выборка: {train_score:.2f}")
    print(f"Тестовая выборка: {test_score:.2f}")

    return model


def make_predictions(model, tomorrow_date):
    # Логика прогнозирования
    pass


if __name__ == "__main__":
    print("Программа запущена")
