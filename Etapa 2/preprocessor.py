

def main_preprocessor():

    import subprocess
    import sys

    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # =============================
    # Librerías principales compatibles
    # =============================
    # Versiones compatibles probadas:
    install("numpy==1.24.4")
    install("scipy==1.10.1")
    install("pandas==2.1.1")
    install("scikit-learn==1.3.2")
    install("imbalanced-learn==0.13.0")
    install("matplotlib==3.7.2")
    install("nltk==3.8.1")
    install("contractions==0.1.73")
    install("inflect==7.5.0")
    install("openai==2.2.0")
    install("ipython==8.15.0")  # para display de DataFrames
    install("jupyter==1.0.0")
    install("seaborn==0.12.2")
   
    
    from sklearn.base import BaseEstimator, TransformerMixin
    import re, unicodedata
    import inflect
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Instalación de librerias
    import pandas as pd
    import numpy as np
    import sys
    import re, string, unicodedata
    import contractions
    import inflect
    from nltk import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import LancasterStemmer, WordNetLemmatizer

    from sklearn.model_selection import train_test_split,GridSearchCV
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.svm import SVC
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.base import BaseEstimator, ClassifierMixin
    import matplotlib.pyplot as plt


    # librería Natural Language Toolkit, usada para trabajar con textos
    import nltk
    # Punkt permite separar un texto en frases.
    nltk.download('punkt')

    # Descarga todas las palabras vacias, es decir, aquellas que no aportan nada al significado del texto
    # ¿Cuales son esas palabras vacías?
    nltk.download('stopwords')

    # Descarga de paquete WordNetLemmatizer, este es usado para encontrar el lema de cada palabra
    # ¿Qué es el lema de una palabra? ¿Qué tan dificil puede ser obtenerlo, piensa en el caso en que tuvieras que escribir la función que realiza esta tarea?
    nltk.download('wordnet')


    # Uso de la libreria pandas para la lectura de archivos
    #ahora descargamos los datos nuevos de la etapa 2!!!
    import os
    path = os.path.join(os.path.dirname(__file__), "Datos_proyecto.xlsx")
    data = pd.read_excel(path)

    #data=pd.read_excel('Datos_proyecto.xlsx')
    # Asignación a una nueva variable de los datos leidos
    data_t=data

    data_t.info()


    from scipy import stats as st

    textos = data_t.copy()
    textos['Conteo'] = [len(x) for x in textos['textos']]
    textos['Max'] = [[max([len(x) for x in i.split(' ')])][0] for i in textos['textos']]
    textos['Min'] = [[min([len(x) for x in i.split(' ')])][0] for i in textos['textos']]
    textos['Media'] = [[np.mean([len(x) for x in i.split(' ')])][0] for i in textos['textos']]



    nltk.download('stopwords')

    stop_words = set(stopwords.words("spanish"))
    def remove_stopwords(words):
        return [w for w in words if w not in stop_words]

    def remove_non_ascii(words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            if word is not None:
                new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
                new_words.append(new_word)
        return new_words

    def to_lowercase(words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
                if word is not None:
                    new_words.append(word.lower())
        return new_words
    def remove_punctuation(words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            if word is not None:
                new_word = re.sub(r'[^\w\s]', '', word)
                if new_word != '':
                    new_words.append(new_word)
        return new_words

    def replace_numbers(words):
    #    """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = inflect.engine()
        print(words)
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
                print("if " + new_word)
            else:
                new_words.append(word)
        return new_words



    def preprocessing(words):
        words = to_lowercase(words)
        words = remove_stopwords(words)
        words = replace_numbers(words)
        words = remove_punctuation(words)
        words = remove_non_ascii(words)
        return words

    data_t['textos'] = data_t['textos'].apply(contractions.fix) #Aplica la corrección de las contracciones

    nltk.download('punkt_tab')

    data_t['words'] = data_t['textos'].apply(word_tokenize)
    data_t.head()

    data_t['words'].dropna()
    data_t.shape
    data_t['words'].info()

    data_t['words1']=data_t['words'].apply(preprocessing) #Aplica la eliminación del ruido

    data_t.head()

    data_t['words'] = data_t['words'].apply(lambda x: ' '.join(map(str, x)))
    data_t

    data_t['clean_text'] = data_t['words1'].apply(lambda x: ' '.join(map(str, x)))
    X_data, y_data = data_t['clean_text'], data_t['labels']

    dummy = CountVectorizer(binary=True)
    X_dummy = dummy.fit_transform(X_data)
    print(X_dummy.shape)
    X_dummy.toarray()[0]

    count = CountVectorizer()
    X_count = count.fit_transform(X_data)
    print(X_count.shape)
    X_count.toarray()[0]

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_tfidf = tfidf.fit_transform(data_t['clean_text'])


    tfidf_df = pd.DataFrame(
        X_tfidf.toarray(),
        columns=tfidf.get_feature_names_out()
    )

    print(tfidf_df[1:])

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=42, stratify=y_data
    )

    # Stopwords y números
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


    return X_train, X_test, y_train, y_test,y_data,X_data,data_t

