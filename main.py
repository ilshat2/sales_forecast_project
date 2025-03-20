import pandas as pd


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


if __name__ == "__main__":
    print("Программа запущена")
