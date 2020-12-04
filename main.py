from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import re
import nltk
from sklearn.datasets import load_files
import pickle
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


class LogisticRegression_(object):
    def __init__(self, n_iter=1000, eta=0.1):
        self.n_iter = n_iter
        self.eta = eta
        self.weights = []
        self.misclass_per_iter = []

    def fit(self, X, y):
        X = self.add_column_with_ones(X)
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.n_iter):
            self.weights -= self.eta * np.dot(X.T, self.costs(X, y)) / len(y)
            self.misclass_per_iter.append(np.absolute(self.costs(X, y)).sum())
            
    def costs(self, X, y):
        return self.hypothesis(X) - y

    def hypothesis(self, X):
        return 1 / (1 + np.exp(-np.dot(X, self.weights)))

    def predict(self, X):
        X = self.add_column_with_ones(X)
        return np.where(self.hypothesis(X) >= 0.5, 1, 0)

    def add_column_with_ones(self, X):
        return np.concatenate([np.ones((len(X), 1)), X], axis=1)


df = pd.read_csv("train.csv")  # red file train.csv
# nltk.download('wordnet')
# nltk.download('stopwords')
en_stopwords = nltk.corpus.stopwords.words('english')

df_test = pd.read_csv("test.csv")
y, x = df["Label"].to_list(), df["TweetText"].to_list()

y = [1 if i == 'Sports' else 0 for i in y]
stemmer = WordNetLemmatizer()

def ft_clean(x):
    texts = []
    for sen in range(0, len(x)):
        text = re.sub(r'\W', ' ', str(x[sen]))
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        text = re.sub(r'^b\s+', '', text)
        text = re.sub(r"\n","",text)  
        text = re.sub(r"\d","",text)       
        text = re.sub(r'[^\x00-\x7f]',r' ',text)
        text = re.sub(r'[^\w\s]','',text) 
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = text.lower()
        text = text.split()
        text = [stemmer.lemmatize(word) for word in text]
        text = [w for w in text if not w in en_stopwords]
        text = ' '.join(text)
        texts.append(text)
    return texts


x = ft_clean(x)

vectorizer = CountVectorizer(
    max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))

x = vectorizer.fit_transform(x).toarray()
tfidfconverter = TfidfTransformer()
x = tfidfconverter.fit_transform(x).toarray()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

classifier = LogisticRegression_()
classifier.fit(np.array(X_train), np.array(y_train))

y_pred = classifier.predict(np.array(X_test))


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

