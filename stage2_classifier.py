import json
import pandas as pd
import language_tool_python
from nrclex import NRCLex
import spacy
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression


# read in
file_path = 'test_output.jsonl'
with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]
df = pd.DataFrame(data)



# label
# df['Label'] = [1,1,0,0,0,0,0,0,0,1]
df['Label'] = [ 1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,
                1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,1,1,
                1,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,1,1,0,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0 ]

# Expand EmotionScores dictionary into separate columns
emotions_df = df['EmotionScores'].apply(pd.Series).fillna(0)
df = pd.concat([df, emotions_df], axis=1)
df = df.drop(columns=['EmotionScores'])

X = df.drop(columns=['Label'])
y = df['Label']                 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)



clf = LogisticRegression(max_iter=10000)  # increase max_iter if convergence warning
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
# print(list(y_pred))  # or use y_pred.tolist()

# Evaluate performance
print(accuracy_score(y_test, y_pred))
# Just print the predicted labels as a list
print(list(y_pred))


# create output file
output_df = pd.DataFrame({
    "index": range(1, len(y_pred) + 1),
    "real_news": ["yes" if pred == 1 else "no" for pred in y_pred]
})
output_df.to_csv("group44_stage2.csv", index=False)

