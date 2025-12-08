import pandas as pd, pickle, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def parse_data():
    true_news = pd.read_csv("dataset/dataset_binary/True.csv")
    fake_news = pd.read_csv("dataset/dataset_binary/Fake.csv")

    true_news['label'] = 1
    fake_news['label'] = 0

    df = pd.concat([true_news, fake_news], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def vectorize():
    df = parse_data()
    
    X = df['text']
    y = df['label']
    
    X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    
    with open("dataset/vectorized_data.pkl", "wb") as f:
        pickle.dump((X_train, y_train, X_test, y_test), f)
    
    print("Vectorized data saved to vectorized_data.pkl")


def load_data():
    if not os.path.exists("dataset/vectorized_data.pkl"):
        vectorize()

    with open("dataset/vectorized_data.pkl", "rb") as f:
        X_train, y_train, X_test, y_test = pickle.load(f)
    
    return X_train, y_train, X_test, y_test