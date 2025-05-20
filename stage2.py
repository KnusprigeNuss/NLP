import json
import pandas as pd
import language_tool_python
from nrclex import NRCLex
import spacy
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
import textstat



############## FUNCTIONS ######################
def semantic_fact_match_score(article_sentences):
    article_embeddings = model.encode(article_sentences, convert_to_tensor=True)

    cos_sim = util.pytorch_cos_sim(article_embeddings, kb_embeddings)

    match_count = (cos_sim > 0.6).any(dim=1).sum().item()
    return match_count


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
with open(file_path, 'r') as f:
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
kb_facts = df_kb["facts"] + df_kb["scientific_data"] + df_kb["prevention"]
kb_embeddings = model.encode(kb_facts, convert_to_tensor=True)

# nlp = spacy.load("en_core_web_sm")
semantic_match_counts = []
fact_match_counts = []

for i in range(len(df)):
    doc = nlp(df['Text'].iloc[i])
    article_sentences = [sent.text for sent in doc.sents]
    fact_match_count_semantic = semantic_fact_match_score(article_sentences)
    semantic_match_counts.append(fact_match_count_semantic)

df['SemanticFactMatchCount'] = semantic_match_counts


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
output_file_path = 'test_output.jsonl'
with open(output_file_path, 'w') as output_file:
    for record in df.to_dict(orient='records'):
        output_file.write(json.dumps(record) + '\n')
