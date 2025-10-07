from sklearn.base import BaseEstimator, TransformerMixin
import re, unicodedata
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = set(stopwords.words("spanish"))
p = inflect.engine()
class TextPreprocessor(BaseEstimator, TransformerMixin):
            def __init__(self, max_features=5000, ngram_range=(1,2)):
                self.max_features = max_features
                self.ngram_range = ngram_range
                # inicializar vectorizador
                self.vectorizer = TfidfVectorizer(max_features=self.max_features,
                                                ngram_range=self.ngram_range)

            def remove_stopwords(self, words):
                return [w for w in words if w not in stop_words]

            def to_lowercase(self, words):
                return [w.lower() for w in words if w is not None]

            def remove_punctuation(self, words):
                return [re.sub(r'[^\w\s]', '', w) for w in words if re.sub(r'[^\w\s]', '', w) != '']

            def remove_non_ascii(self, words):
                new_words = []
                for word in words:
                    new_word = unicodedata.normalize('NFKD', word).encode('ascii','ignore').decode('utf-8','ignore')
                    if new_word != "":
                        new_words.append(new_word)
                return new_words

            def replace_numbers(self, words):
                new_words = []
                for word in words:
                    if word.isdigit():
                        new_word = p.number_to_words(word)
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

                return " ".join(words)  # devolvemos texto plano listo para TF-IDF

            def fit(self, X, y=None):
                cleaned = [self.preprocessing(text) for text in X]
                self.vectorizer = TfidfVectorizer(max_features=self.max_features,
                                                ngram_range=self.ngram_range)
                self.vectorizer.fit(cleaned)
                return self

            def transform(self, X, y=None):
                cleaned = [self.preprocessing(text) for text in X]
                return self.vectorizer.transform(cleaned)