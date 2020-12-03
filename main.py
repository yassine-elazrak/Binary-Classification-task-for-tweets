import numpy as np
import re
import nltk
from sklearn.datasets import load_files
# nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# vectorizer = CountVectorizer()
df = pd.read_csv("train.csv")  # red file train.csv
# nltk.download('stopwords')
# 
# nltk.download('wordnet')
y, x = df["Label"].to_list(), df["TweetText"].to_list()
en_stopwords = nltk.corpus.stopwords.words('english')

stemmer = WordNetLemmatizer()

# print("====\n\nx(0)",x[0])
def ft_Stem(x):
    documents = []
    for sen in range(0, len(x)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(x[sen]))
    # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
    # Converting to Lowercase
        document = document.lower()
    # Lemmatization
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document =[w for w in document if not w in en_stopwords]
        document = ' '.join(document)
        documents.append(document)
    return documents
x = ft_Stem(x)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
x = vectorizer.fit_transform(x).toarray()
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
x = tfidfconverter.fit_transform(x).toarray()
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(x, y)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# y_pred =["'#SecKerry: The value of the @StateDept and @USAID is measured, not in dollars, but in terms of our deepest American values.'"]
# y_pred = ft_Stem(y_pred)
# x = y_pred
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer()
# x = vectorizer.fit_transform(x).toarray()
# from sklearn.feature_extraction.text import TfidfTransformer
# tfidfconverter = TfidfTransformer()
# x = tfidfconverter.fit_transform(x).toarray()
# y_pred = x
y_pred = classifier.predict([x[2]])
print("\n\ny_pred\n\n",y_pred)

# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print(accuracy_score(y_test, y_pred))

# print("nnnn",len(x[0]))
# for i in range(len(x[0])):
#     if x[0][i] != 0:
#         print("=====\n\nnew x(0)",x[0][i])
