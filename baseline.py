import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#use big stopwords libary
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
from nltk.stem import WordNetLemmatizer
import json

nltk.download('wordnet')

with open("data/without_assessment.jsonl", 'r') as file:
    data = [json.loads(line) for line in file]
df = pd.DataFrame(data)
#our self labeled gournd truth
df['Label'] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
                   1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                   1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
                   1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1,
                   1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1,
                   0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]

X = df.drop(columns=['Label'])
y = df['Label']



#we only need Text
df = df.dropna(subset=['Text'])
#all to lower case and remove stopwords
df['Text'] = df['Text'].str.lower()
stop_words_nltk = set(stopwords.words('english'))
stop_words_sklearn = set(ENGLISH_STOP_WORDS)
#we use both stopword libaries
stop_words = stop_words_nltk.union(stop_words_sklearn)

def clean_text(text):
    #we remove punctuation and stopwords
    text = text.translate(str.maketrans('', '', string.punctuation+"“"))
    filtered_words = [word for word in text.split() if word not in stop_words]
    return ' '.join(filtered_words)
def lemmatize_text(text):
    ret_text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return ret_text
print("Text before punctuation and stopwords removable:")
print(df['Text'].head(2))
df['Text'] = df['Text'].apply(clean_text)
#eg: action instead of actions
lemmatizer = WordNetLemmatizer()
df['Text'] = df['Text'].apply(lemmatize_text)
print("The same after the lemmatizer, punctuation and stopwords removable:")
print(df['Text'].head(2))

X = df[['Text']]
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    stop_words=None,
    max_df=0.8,
    min_df=0.005,
    ngram_range=(1, 2),
)
X_train_tfidf = vectorizer.fit_transform(X_train['Text'])
X_test_tfidf = vectorizer.transform(X_test['Text'])

print("Vocabulary size:", len(vectorizer.vocabulary_))

X_true = X_train_tfidf[y_train.values == 1]
X_false = X_train_tfidf[y_train.values == 0]

true_centroid = np.asarray(X_true.mean(axis=0)).flatten()
false_centroid = np.asarray(X_false.mean(axis=0)).flatten()

dist_to_true = cosine_distances(X_test_tfidf, [true_centroid])
dist_to_false = cosine_distances(X_test_tfidf, [false_centroid])

y_pred = (dist_to_true < dist_to_false).astype(int).flatten()

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))