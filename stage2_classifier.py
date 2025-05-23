import json
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, cross_val_score
from catboost import CatBoostClassifier


# read in
file_path = 'test_output_mr_msmarco-distilbert-base-v4.jsonl'
with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]
df = pd.DataFrame(data)


# label solution
df['Label'] = [ 1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,
                1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,1,1,
                1,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,1,1,0,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0 ]


# split emotions col and drop useless ones
emotions_df = df['EmotionScores'].apply(pd.Series).fillna(0)
df = pd.concat([df, emotions_df], axis=1)
df = df.drop(columns=['EmotionScores', 'Index', 'joy'])#,"SemanticFactMatchCount-50",'SemanticFactMatchCount-75','SemanticFactMatchCount-85','SemanticFactMatchCount-80',"SemanticFactMatchCount-70","SemanticFactMatchCount-60",'SemanticFactMatchCount-90',"SemanticFactMatchCount-55","SemanticFactMatchCount-65"])

#use the features found by featuresearch.py
features=['Grammar_errors', 'TotalEntities', 'SemanticFactMatchCount-50', 'SemanticFactMatchCount-55', 'SemanticFactMatchCount-60', 'NOUN', 'VERB', 'ADJ', 'ADV', 'flesch_reading_ease', 'smog_index', 'automated_readability_index', 'words_per_sentence', 'difficult_words', 'RedFlagWords', 'anticipation', 'fear', 'negative', 'positive', 'anger', 'trust', 'sadness']

X = df[features]
y = df['Label']   
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=654,stratify=y)

# random forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
#grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
#grid.fit(X_train, y_train)
#y_pred = grid.predict(X_test)
#print("Best parameters:", grid.best_params_)
#print("Accuracy:", accuracy_score(y_test, y_pred))

#beacuse of overfitting introduce limitations like max_depth
clf = RandomForestClassifier(
    n_estimators=90,
    random_state=420,
    max_depth=60,
    min_samples_split=18,
 )

clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_pred = clf.predict(X_test)


# logistic regression (worse)
# param_grid = {
#     'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50],
#     'penalty': ['l2'],
#     'solver': ['lbfgs', 'saga'],
#     'max_iter': [1000, 5000, 10000]
# }
# grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, n_jobs=-1)
# grid.fit(X_train, y_train)
# y_pred = grid.predict(X_test)
# print("Best parameters:", grid.best_params_)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

importances = clf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Show top features
print(importance_df)

# output
print("train:", accuracy_score(y_train, y_train_pred))
print("test:",accuracy_score(y_test, y_pred))
#cross val because of overfitting problems
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_scaled, y, cv=cv)
print("CV scores:", scores)
print("Mean CV accuracy:", scores.mean())
output_df = pd.DataFrame({
    "index": range(1, len(y_pred) + 1),
    "real_news": ["yes" if pred == 1 else "no" for pred in y_pred]
})
output_df.to_csv("group44_stage2_mr.csv", index=False)


