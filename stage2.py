import json
import pandas as pd
import language_tool_python
from nrclex import NRCLex

# read in
file_path = 'test.jsonl'
with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]
df = pd.DataFrame(data)

# spelling errors (before lowering since that is counted as error)
tool = language_tool_python.LanguageTool('en-US')

def count_errors(text):
    matches = tool.check(text)
    return len(matches)

df['Grammar_errors'] = df['Text'].apply(count_errors)

# preprocessing
df['Title'] = df['Title'].str.lower()
df['Text'] = df['Text'].str.lower()


# emotions
def get_emotions(text):
    emotion = NRCLex(text)
    return emotion.raw_emotion_scores

df['EmotionScores'] = df['Text'].apply(get_emotions)



df = df.drop(columns=['Text', 'Title'])
output_file_path = 'test_output.jsonl'
with open(output_file_path, 'w') as output_file:
    for record in df.to_dict(orient='records'):
        output_file.write(json.dumps(record) + '\n')