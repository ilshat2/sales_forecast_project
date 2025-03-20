import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb


# Загрузка данных
def load_data():
    id1 = pd.read_csv('../id_597.csv')
    id2 = pd.read_csv('../id_16.csv')
    id3 = pd.read_csv('../id_34.csv')
    return pd.concat([id1, id2, id3])


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


if __name__ == "__main__":
    print("Программа запущена")
