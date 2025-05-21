import json
import pandas as pd
import language_tool_python
from nrclex import NRCLex
import spacy
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
import textstat



############## FUNCTIONS ######################
"""def semantic_fact_match_score(article_sentences):
    article_embeddings = model.encode(article_sentences, convert_to_tensor=True)

    cos_sim = util.pytorch_cos_sim(article_embeddings, kb_embeddings)

    match_count = (cos_sim > 0.6).any(dim=1).sum().item()
    return match_count
"""

thresholds_to_test = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

def semantic_fact_match_score(article_sentences, thresholds):
    """
    Calculates semantic fact match counts for a list of article sentences
    against the knowledge base embeddings for multiple cosine similarity thresholds,
    returning results in a specific order as a list.

    Args:
        article_sentences (list): A list of sentences from an article.
        thresholds (list): An ordered list of cosine similarity thresholds to evaluate.

    Returns:
        list: A list of match counts, ordered according to the `thresholds` input.
    """
    if not article_sentences:
        return [0] * len(thresholds) # Return a list of zeros if no sentences

    article_embeddings = model.encode(article_sentences, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(article_embeddings, kb_embeddings)

    max_similarities = cos_sim.max(dim=1).values

    # Collect match counts in the order of the thresholds_to_test list
    ordered_match_counts = []
    for threshold in thresholds:
        match_count = (max_similarities > threshold).sum().item()
        ordered_match_counts.append(match_count)

    return ordered_match_counts
def count_trusted_entities(text, kb_entities):
    doc = nlp(text)
    trusted = 0
    total = 0
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON"]:
            total += 1
            if ent.text in kb_entities["organizations"] or ent.text in kb_entities["scientists"]:
                trusted += 1
    return trusted, total


def count_errors(text, tool):
    matches = tool.check(text)
    return len(matches)


def get_emotions(text):
    emotion = NRCLex(text)
    return emotion.raw_emotion_scores


def get_readability(text):
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "smog_index": textstat.smog_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "words_per_sentence": textstat.words_per_sentence(text),
        "difficult_words": textstat.difficult_words(text)
    }


def pos_distribution(text):
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


def count_red_flags(text):
    return sum(text.lower().count(word) for word in suspicious_keywords)
#######################################################

# read in
file_path = 'without_assessment.jsonl'
# file_path = 'test.jsonl'
with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]
df = pd.DataFrame(data)
df2 = df.copy()
# tolower for the text checker
df2['Title'] = df2['Title'].str.lower()
df2['Text'] = df2['Text'].str.lower()

file_path = 'knowledge_base.jsonl'
with open(file_path, 'r',encoding='utf-8') as f:
    df_kb = json.load(f)


# spelling errors
tool = language_tool_python.LanguageTool('en-US')
df['Grammar_errors'] = df['Text'].apply(count_errors, tool=tool)

# emotions
df['EmotionScores'] = df['Text'].apply(get_emotions)

# trusted entity counting
nlp = spacy.load("en_core_web_sm")
trusted_counts = []
total_counts = []
entity_ratios = []

for i in range(len(df)):
    doc = nlp(df['Text'].iloc[i])
    trusted, total = count_trusted_entities(doc, df_kb["trusted_entities"])
    entity_ratio = trusted / total if total > 0 else 0
    trusted_counts.append(trusted)
    total_counts.append(total)
    entity_ratios.append(entity_ratio)

df['TrustedEntities'] = trusted_counts
df['TotalEntities'] = total_counts
df['EntityRatio'] = entity_ratios



# check facts of the knowledge base
# TODO: not really working yet!!
model = SentenceTransformer('all-MiniLM-L6-v2')
kb_facts = df_kb["facts"] + df_kb["scientific_data"] + df_kb["prevention"]+df_kb["support"]
kb_embeddings = model.encode(kb_facts, convert_to_tensor=True)

# nlp = spacy.load("en_core_web_sm")
semantic_match_counts_50 = []
semantic_match_counts_55 = []
semantic_match_counts_60 = []
semantic_match_counts_65 = []
semantic_match_counts_70 = []
semantic_match_counts_75 = []
semantic_match_counts_80 = []
semantic_match_counts_85 = []
semantic_match_counts_90 = []
fact_match_counts = []

for i in range(len(df)):
    doc = nlp(df['Text'].iloc[i])
    article_sentences = [sent.text for sent in doc.sents]
    match_count50,match_count55,match_count60,match_count65,match_count70,match_count75,match_count80,match_count85,match_count90 = semantic_fact_match_score(article_sentences,thresholds=thresholds_to_test)
    semantic_match_counts_50.append(match_count50)
    semantic_match_counts_55.append(match_count55)
    semantic_match_counts_60.append(match_count60)
    semantic_match_counts_65.append(match_count65)
    semantic_match_counts_70.append(match_count70)
    semantic_match_counts_75.append(match_count75)
    semantic_match_counts_80.append(match_count80)
    semantic_match_counts_85.append(match_count85)
    semantic_match_counts_90.append(match_count90)
    print("Document:",i," with match count @ 0.6 ",match_count60)

df['SemanticFactMatchCount-50'] = semantic_match_counts_50
df['SemanticFactMatchCount-55'] = semantic_match_counts_55
df['SemanticFactMatchCount-60'] = semantic_match_counts_60
df['SemanticFactMatchCount-65'] = semantic_match_counts_65
df['SemanticFactMatchCount-70'] = semantic_match_counts_70
df['SemanticFactMatchCount-75'] = semantic_match_counts_75
df['SemanticFactMatchCount-80'] = semantic_match_counts_80
df['SemanticFactMatchCount-85'] = semantic_match_counts_85
df['SemanticFactMatchCount-90'] = semantic_match_counts_90

# POS Tagging
pos_df = df['Text'].apply(pos_distribution).apply(pd.Series)
df = pd.concat([df, pos_df], axis=1)


# readability
readability_df = df['Text'].apply(get_readability).apply(pd.Series)
df = pd.concat([df, readability_df], axis=1)


# redflagwords
suspicious_keywords = ["hoax", "exposed", "globalist", "scam", "shocking", "hidden", "fake", "alarmist", "agenda"]
df['RedFlagWords'] = df['Text'].apply(count_red_flags)


# output
df = df.drop(columns=['Text', 'Title'])
output_file_path = 'test_output_mr_msmarco-distilbert-base-v4.jsonl'
with open(output_file_path, 'w') as output_file:
    for record in df.to_dict(orient='records'):
        output_file.write(json.dumps(record) + '\n')
