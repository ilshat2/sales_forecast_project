import pandas as pd
import numpy as np
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


def predict_tomorrow(model, df, le_dish, le_rest):
    """Прогнозирование продаж на завтра"""

    # Определение завтрашней даты
    last_date = df['OpenDate.Typed'].max()
    tomorrow = last_date + pd.DateOffset(days=1)

    # Создание шаблона прогноза
    forecast_data = []
    for dish in df['DishName'].unique():
        for rest in df['RestorauntGroup'].unique():
            # Исторические данные для лагов
            history = df[
                (df['DishName'] == dish) & 
                (df['RestorauntGroup'] == rest)
            ]
            lag3_value = history.tail(3)['DishAmountInt'].mean()

            forecast_data.append({
                'DishEncoded': le_dish.transform([dish])[0],
                'RestEncoded': le_rest.transform([rest])[0],
                'DayOfWeek': tomorrow.dayofweek,
                'Month': tomorrow.month,
                'CloseHour': 18,  # Предполагаем пиковое время
                'IsWeekend': 1 if tomorrow.dayofweek in [5, 6] else 0,
                'Lag3': lag3_value if not np.isnan(lag3_value) else 0,
                'DishDiscountSumInt': 0  # Без скидок по умолчанию
            })

    # Прогнозирование
    forecast_df = pd.DataFrame(forecast_data)
    predictions = model.predict(forecast_df)

    # Форматирование результатов
    result = pd.DataFrame({
        'DishName': le_dish.inverse_transform(forecast_df['DishEncoded']),
        'RestorauntGroup': le_rest.inverse_transform(forecast_df['RestEncoded']),
        'PredictedAmount': np.round(predictions).astype(int)
    })

    return result[result['PredictedAmount'] > 0]


if __name__ == "__main__":
    print("Программа запущена")
