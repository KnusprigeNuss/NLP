import json
import pandas as pd
import language_tool_python
from nrclex import NRCLex
import spacy


# read in
file_path = 'test.jsonl'
with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]
df = pd.DataFrame(data)

file_path = 'knowledge_base.jsonl'
with open(file_path, 'r') as f:
    df_kb = json.load(f)


# # preprocessing (destroys the accuracy of spelling error check)
# df['Title'] = df['Title'].str.lower()
# df['Text'] = df['Text'].str.lower()


############# SPELLING ERRORS #############################
# tool = language_tool_python.LanguageTool('en-US')

# def count_errors(text):
#     matches = tool.check(text)
#     return len(matches)

# df['Grammar_errors'] = df['Text'].apply(count_errors)


############# EMOTION #############################
# def get_emotions(text):
#     emotion = NRCLex(text)
#     return emotion.raw_emotion_scores

# df['EmotionScores'] = df['Text'].apply(get_emotions)

# df = df.drop(columns=['Text', 'Title'])
# output_file_path = 'test_output.jsonl'
# with open(output_file_path, 'w') as output_file:
#     for record in df.to_dict(orient='records'):
#         output_file.write(json.dumps(record) + '\n')


############# TRUSTED ENTITY COUNTING #############
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

nlp = spacy.load("en_core_web_sm")
for i in range(len(df)):
    doc = nlp(df['Text'].iloc[i])
    trusted, total = count_trusted_entities(doc, df_kb["trusted_entities"])
    entity_ratio = trusted / total if total > 0 else 0
    print(f"Trusted entities: {trusted}, Total entities: {total}, Ratio: {entity_ratio:.2f}")




