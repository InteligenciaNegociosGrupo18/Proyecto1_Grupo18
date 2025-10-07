import joblib
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata
import inflect

nltk.download('stopwords')
nltk.download('punkt')

class TextPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(max_features=self.max_features,
                                          ngram_range=self.ngram_range)
        self.stop_words = set(stopwords.words("spanish"))
        self.p = inflect.engine()

    def remove_stopwords(self, words):
        return [w for w in words if w not in self.stop_words]

    def to_lowercase(self, words):
        return [w.lower() for w in words if w is not None]

    def remove_punctuation(self, words):
        return [re.sub(r'[^\w\s]', '', w) for w in words if re.sub(r'[^\w\s]', '', w) != '']

    def remove_non_ascii(self, words):
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            if new_word != "":
                new_words.append(new_word)
        return new_words

    def replace_numbers(self, words):
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = self.p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    def preprocessing(self, text):
        if text is None or str(text).strip() == "" or str(text).lower() == "nan":
            return ""
        words = word_tokenize(str(text))
        words = self.to_lowercase(words)
        words = self.remove_stopwords(words)
        words = self.replace_numbers(words)
        words = self.remove_punctuation(words)
        words = self.remove_non_ascii(words)
        return " ".join(words)

    def fit(self, X, y=None):
        cleaned = [self.preprocessing(text) for text in X]
        self.vectorizer.fit(cleaned)
        return self

    def transform(self, X, y=None):
        cleaned = [self.preprocessing(text) for text in X]
        return self.vectorizer.transform(cleaned)



def build_pipeline():

    pipeline = Pipeline([
        ('vectorizer', TextPreprocessor(max_features=5000, ngram_range=(1, 2))),
        ('model', RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42
        ))
    ])
    return pipeline



if __name__ == "__main__":
    import os
    path = os.path.join(os.path.dirname(__file__), "Datos_etapa 2.xlsx")
    data = pd.read_excel(path)
    data = data.dropna(subset=['textos', 'labels']).drop_duplicates()

    X = data['textos']
    y = data['labels']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

   
    pipeline = build_pipeline()
    print("Entrenando modelo...")
    pipeline.fit(X_train, y_train)

    
    preds = pipeline.predict(X_test)
    print("\n=== Reporte de Clasificación ===")
    print(classification_report(y_test, preds))
    print("\n=== Matriz de Confusión ===")
    print(confusion_matrix(y_test, preds))

    
    import os
    model_path = os.path.join(os.path.dirname(__file__), "pipeline_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\nModelo guardado exitosamente en: {model_path}")

