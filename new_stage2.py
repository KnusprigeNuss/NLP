import json
import pandas as pd
import language_tool_python
from nrclex import NRCLex
import spacy
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
import textstat
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def feature_extraction_dataset(src_path, dest_path):
    def count_trusted_entities(text):
        doc = nlp(text)
        trusted = 0
        total = 0
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON"]:
                total += 1
                if ent.text in df_kb["trusted_entities"]["organizations"] or ent.text in df_kb["trusted_entities"][
                    "scientists"]:
                    trusted += 1
        return trusted, total

    def semantic_fact_match_score(article_sentences):
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        model = SentenceTransformer('msmarco-distilbert-base-v4')
        #model = SentenceTransformer('all-MiniLM-L6-v2')
        #facts from the knowlage base
        kb_facts = df_kb["facts"] + df_kb["scientific_data"] + df_kb["prevention"] + df_kb["support"]
        kb_embeddings = model.encode(kb_facts, convert_to_tensor=True)
        article_embeddings = model.encode(article_sentences, convert_to_tensor=True)
        cos_sim = util.pytorch_cos_sim(article_embeddings, kb_embeddings)
        max_similarities = cos_sim.max(dim=1).values
        #collect matches for each threshold
        matches_count = []
        for threshold in thresholds:
            match_count = (max_similarities > threshold).sum().item()
            matches_count.append(match_count)
        return matches_count

    #Spelling errors
    def count_errors(text, tool):
        matches = tool.check(text)
        return len(matches)

    #Extract emotions
    def get_emotions(text):
        emotion = NRCLex(text)
        return emotion.raw_emotion_scores

    #Extract text readability
    def get_readability(text):
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "smog_index": textstat.smog_index(text),
            "automated_readability_index": textstat.automated_readability_index(text),
            "words_per_sentence": textstat.words_per_sentence(text),
            "difficult_words": textstat.difficult_words(text)
        }

    #Extract noun,verb,adj,adv
    def grammatic_distribution(text):
        doc = nlp(text)
        pos_counts = {
            "NOUN": 0,
            "VERB": 0,
            "ADJ": 0,
            "ADV": 0
        }
        for token in doc:
            if token.pos_ in pos_counts:
                pos_counts[token.pos_] += 1
        return pos_counts

    #Count blacklisted words
    def count_blacklist_words(text):
        suspicious_keywords = ["hoax", "exposed", "globalist", "scam", "shocking", "hidden", "fake", "alarmist",
                               "agenda"]
        return sum(text.lower().count(word) for word in suspicious_keywords)

    #Knowlege Base
    file_path = 'knowledge_base.jsonl'
    with open(file_path, 'r', encoding='utf-8') as f:
        df_kb = json.load(f)
    #Preporcessing dataset
    with open(src_path, 'r') as file:
        data = [json.loads(line) for line in file]
    df = pd.DataFrame(data)
    df2 = df.copy()
    print("\n-------------Dataset-------------\n")
    print(df2.head())
    #ToLower for the text checker
    df2['Title'] = df2['Title'].str.lower()
    df2['Text'] = df2['Text'].str.lower()
    #Check for spelling errores
    tool = language_tool_python.LanguageTool('en-US')
    df['Grammar_errors'] = df['Text'].apply(count_errors, tool=tool)
    print("\n-------------Grammar errors--------\n")
    print(df['Grammar_errors'].head())
    #Extract emotions
    df['EmotionScores'] = df['Text'].apply(get_emotions)
    print("\n-------------Emotion scores--------\n")
    print(df['EmotionScores'].head())
    #Trusted entity counting
    nlp = spacy.load("en_core_web_sm")
    trusted_counts = []
    total_counts = []
    entity_ratios = []
    for i in range(len(df)):
        doc = nlp(df['Text'].iloc[i])
        trusted, total = count_trusted_entities(doc)
        entity_ratio = trusted / total if total > 0 else 0
        trusted_counts.append(trusted)
        total_counts.append(total)
        entity_ratios.append(entity_ratio)

    df['TrustedEntities'] = trusted_counts
    df['TotalEntities'] = total_counts
    df['EntityRatio'] = entity_ratios
    print("\n-------------Trusted entities--------\n")
    print(df['TrustedEntities'].head())
    print("\n-------------Total entities----------\n")
    print(df['TotalEntities'].head())
    print("\n-------------Entity ratio------------\n")
    print(df['EntityRatio'].head())
    #Cos-Sim of knowlagebase and the sentences
    semantic_match_counts_50 = []
    semantic_match_counts_55 = []
    semantic_match_counts_60 = []
    semantic_match_counts_65 = []
    semantic_match_counts_70 = []
    semantic_match_counts_75 = []
    semantic_match_counts_80 = []
    semantic_match_counts_85 = []
    semantic_match_counts_90 = []
    for i in range(len(df)):
        doc = nlp(df['Text'].iloc[i])
        article_sentences = [sent.text for sent in doc.sents]
        match_count50, match_count55, match_count60, match_count65, match_count70, match_count75, match_count80, match_count85, match_count90 = semantic_fact_match_score(
            article_sentences)
        print(i, match_count55)
        semantic_match_counts_50.append(match_count50)
        semantic_match_counts_55.append(match_count55)
        semantic_match_counts_60.append(match_count60)
        semantic_match_counts_65.append(match_count65)
        semantic_match_counts_70.append(match_count70)
        semantic_match_counts_75.append(match_count75)
        semantic_match_counts_80.append(match_count80)
        semantic_match_counts_85.append(match_count85)
        semantic_match_counts_90.append(match_count90)
    df['SemanticFactMatchCount-50'] = semantic_match_counts_50
    df['SemanticFactMatchCount-55'] = semantic_match_counts_55
    df['SemanticFactMatchCount-60'] = semantic_match_counts_60
    df['SemanticFactMatchCount-65'] = semantic_match_counts_65
    df['SemanticFactMatchCount-70'] = semantic_match_counts_70
    df['SemanticFactMatchCount-75'] = semantic_match_counts_75
    df['SemanticFactMatchCount-80'] = semantic_match_counts_80
    df['SemanticFactMatchCount-85'] = semantic_match_counts_85
    df['SemanticFactMatchCount-90'] = semantic_match_counts_90
    print("\n-------------SemanticFactMatchCount @55--\n")
    print(df['SemanticFactMatchCount-55'].head())
    #Grammatic distibution(POS Tagging)
    pos_df = df['Text'].apply(grammatic_distribution).apply(pd.Series)
    df = pd.concat([df, pos_df], axis=1)
    print("\n-------------POS tagging-----------------\n")
    print(pos_df.head())
    #Check Readability
    readability_df = df['Text'].apply(get_readability).apply(pd.Series)
    df = pd.concat([df, readability_df], axis=1)
    print("\n-------------Readability-----------------\n")
    print(readability_df.head())
    #Blacklisted words
    df['RedFlagWords'] = df['Text'].apply(count_blacklist_words)
    print("\n-------------Blacklisted words-----------\n")
    print(df['RedFlagWords'].head())
    #Save only the extracted features
    df = df.drop(columns=['Text', 'Title'])
    with open(dest_path, 'w') as output_file:
        for record in df.to_dict(orient='records'):
            output_file.write(json.dumps(record) + '\n')


def find_best_features_forest(feature_src):
    with open(feature_src, 'r') as file:
        data = [json.loads(line) for line in file]
    df = pd.DataFrame(data)
    #our self labeled gournd truth
    df['Label'] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
                   1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                   1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
                   1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1,
                   1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1,
                   0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]
    #just preprocessing - EmotionScores is in an other format
    emotions_df = df['EmotionScores'].apply(pd.Series).fillna(0)
    df = pd.concat([df, emotions_df], axis=1)
    df = df.drop(columns=['EmotionScores', 'Index','joy'])

    X = df.drop(columns=['Label'])
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    #Test logistic regression with recursive feature elimination
    lr = LogisticRegression(max_iter=2000, solver='saga')
    rfecv_lr = RFECV(
        estimator=lr,
        step=1,
        cv=StratifiedKFold(5),
        scoring='accuracy'
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', rfecv_lr),
        ('classifier', lr)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    sel_feature_maks = pipeline.named_steps['feature_selection'].support_
    selected_features = X.columns[sel_feature_maks]
    print("Logistic regression accuracy:", accuracy_score(y_test, y_pred))
    print("Optimal number of features:", rfecv_lr.n_features_)
    print("Selected features:", list(selected_features))
    #Test Randomforest with recursive feature elimination
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rfecv_rf = RFECV(
        estimator=rf,
        step=1,
        cv=StratifiedKFold(5),
        scoring='accuracy'
    )

    # Fit RFECV
    rfecv_rf.fit(X_train_scaled, y_train)

    # Transform datasets
    X_train_rfe = rfecv_rf.transform(X_train_scaled)
    X_test_rfe = rfecv_rf.transform(X_test_scaled)

    # Train final model on selected features
    clf = RandomForestClassifier(n_estimators=100, random_state=423)
    clf.fit(X_train_rfe, y_train)
    y_pred = clf.predict(X_test_rfe)
    selected_features = X.columns[rfecv_rf.support_]

    print("Random Forest accuracy:", accuracy_score(y_test, y_pred))
    print("Optimal number of features:", rfecv_rf.n_features_)
    print("Selected features:", list(selected_features))

def label_data(feature_src):
    #from find_best_feature_forest()
    selected_features=['Grammar_errors', 'TotalEntities', 'SemanticFactMatchCount-50', 'SemanticFactMatchCount-55', 'SemanticFactMatchCount-60', 'NOUN', 'VERB', 'ADJ', 'ADV', 'flesch_reading_ease', 'smog_index', 'automated_readability_index', 'words_per_sentence', 'difficult_words', 'RedFlagWords', 'anticipation', 'fear', 'negative', 'positive', 'anger', 'trust', 'sadness']
    with open(feature_src, 'r') as file:
        data = [json.loads(line) for line in file]
    df = pd.DataFrame(data)
    #our self labeled gournd truth
    df['Label'] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
                   1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                   1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
                   1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1,
                   1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1,
                   0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]
    #just preprocessing - EmotionScores is in an other format
    emotions_df = df['EmotionScores'].apply(pd.Series).fillna(0)
    df = pd.concat([df, emotions_df], axis=1)
    df = df.drop(columns=['EmotionScores', 'Index','joy'])

    X = df[selected_features]
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
def main():
    feature_extraction = False
    find_best_features = True
    training = False
    src_path = 'data/without_assessment.jsonl'
    dest_path = 'test_output_mr_msmarco-distilbert-base-v4.jsonl'
    print("***************NLP Stage 2***************")
    if feature_extraction:
        print("#####################Feature Extraction#####################")
        feature_extraction_dataset(src_path, dest_path)
    if find_best_features:
        print("#####################Find best features#####################")
        find_best_features_forest(dest_path)
        #TODO include hyperparam search for random forest
    if label_data:
        label_data(dest_path)



if __name__ == "__main__":
    main()
