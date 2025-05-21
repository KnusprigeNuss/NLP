import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel


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


X = df.drop(columns=['Label'])
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use feature selection model with RandomForest
selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")
selector.fit(X_train_scaled, y_train)

# Transform datasets to keep only selected features
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Train on reduced feature set
clf = RandomForestClassifier(n_estimators=100, random_state=423)
clf.fit(X_train_selected, y_train)
y_pred = clf.predict(X_test_selected)

# Accuracy after feature selection
print("Accuracy (selected features):", accuracy_score(y_test, y_pred))

# See which features were selected
selected_mask = selector.get_support()
selected_features = X.columns[selected_mask]
print("Selected features:", list(selected_features))