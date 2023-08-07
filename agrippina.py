import requests
import json
import re
from collections import defaultdict
import openai
import pandas as pd

openai.api_key = "[YOUR_API_KEY]"

def generate_questions(h1, h2, passage):
    completion = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=[
                {"role": "user", "content": '''
            Consider the following passage from the wikpedia article on Agrippina, %s, %s:
            ---
            %s
            ---
            Generate 20 prompts and completions pairs that would teach a davinci GPT3 model the content of this passage. 
            Prompts should be complete questions.
            Completions should contain plenty of context so davinci can understand the flow of events, character motivations, and relationships.
            Prompts and completions should be long and detailed. 
            Reply in JSONL format
                ''' % (h1, h2, passage)},
              ]
            )
    return completion

def generate_questions_basic(h1, h2, passage):
    completion = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=[
                {"role": "user", "content": '''
            Consider the following passage from the wikpedia article on Agrippina, %s, %s:
            ---
            %s
            ---
            Generate 20 prompts and completions pairs that would teach a davinci GPT3 model the content of this passage. 
            Reply in JSONL format
                ''' % (h1, h2, passage)},
              ]
            )
    return completion


def remove_tags(string, tag):
    toks = string.split(f'<{tag}')
    new_toks = []
    for tok in toks:
        new_toks.append(tok.split(f'</{tag}>')[-1])
    return ''.join(new_toks)


if __name__ == '__main__':
    res_json = requests.get('https://en.wikipedia.org/w/api.php', params={
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": "Agrippina_(opera)",
        "formatversion": "2",
        "rvprop": "content",
        "rvslots": "*"
    }).json()
    processed = re.sub(r'\[\[File:[^\n]+', '', res_json['query']['pages'][0]['revisions'][0]['slots']['main']['content'])
    processed = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', processed)
    processed = remove_tags(processed, 'ref')
    processed = remove_tags(processed, 'blockquote')
    processed = processed.replace('[[', '').replace(']]', '')
    processed = re.sub(r'\{\{[^\}]+\}\}', r'', processed)
    processed = processed.split('== References ==')[0]
    processed = re.sub(r'\'{2}', '', processed)
    hierarchy_1 = 'Introduction'
    hierarchy_2 = 'Main'
    hierarchical_data = defaultdict(lambda: defaultdict(list))

    for paragraph in processed.split('\n'):
        if paragraph == '':
            continue
        if paragraph.startswith('==='):
            hierarchy_2 = paragraph.split('===')[1]
        elif paragraph.startswith('=='):
            hierarchy_1 = paragraph.split('==')[1]
            hierarchy_2 = 'Main'
        else:
            print(hierarchy_1, hierarchy_2)
            hierarchical_data[hierarchy_1][hierarchy_2].append(paragraph)

    questions = defaultdict(lambda: defaultdict(list))
    for h_1, h1_data in hierarchical_data.items():
        if h_1 != 'Synopsis':
            continue
        for h_2, h2_data in h1_data.items():
            print('==========', h_1, h_2, '===========')
            passage = '\n\n'.join(h2_data)
            prompts_completion = generate_questions(h_1, h_2, passage)
            prompts_completion_basic = generate_questions_basic(h_1, h_2, passage)

            print(passage)
            questions[h_1][h_2] = {
                'passage': passage,
                'prompts_completion': prompts_completion,
                'prompts_completion_basic': prompts_completion_basic
            }
            print(prompts_completion.choices[0]['message']['content'])
            print(prompts_completion_basic.choices[0]['message']['content'])

    all_questions = []
    for h1, h1_data in questions.items():
        for h2, h2_data in h1_data.items():
            for key in ['prompts_completion', 'prompts_completion_basic']:
                for ob in h2_data[key].choices[0]['message']['content'].split('\n'):
                    try:
                        js = json.loads(ob)
                        js['h1'] = h1
                        js['h2'] = h2
                        all_questions.append(js)
                    except Exception:
                        print(ob)
    df = pd.DataFrame(all_questions)
    df['prompt'] = df.apply(
        lambda row: 'Answer the following question about the Opera Agrippina, Section %s, subsection %s: \n %s \n ### \n'  % (
            row['h1'], row['h2'], row['prompt']
        ), axis=1)            
    df['completion'] = df['completion'].map(lambda x: f'{x} [DONE]')
    with open('agrippina_training.jsonl', 'w') as fp_agrippina:
        fp_agrippina.write(df[['prompt', 'completion']].to_json(orient='records', lines=True))

    # run the training data here
    # export OPENAI_API_KEY=[OPENAI_API_KEY]
    # openai api fine_tunes.create -t agrippina_training.jsonl -m davinci --suffix agrippina
    model_name = '[YOUR_MODEL_NAME]'

    prompt = "Answer the following question about the Opera Agrippina: \n Who does Agrippina plot to secure the throne for? \n ### \n"

    res = requests.post('https://api.openai.com/v1/completions', headers={
        'Content-Type': 'application/json',
        'Authorization': 'Bearer [OPENAI_API_KEY]'
    }, json={
        'prompt': prompt,
        'max_tokens': 500,
        'model': model_name,
        'stop': '[DONE]'
    })

    print(res.json()['choices'][0]['text']) # should print "Agrippina plots to secure the throne for Nero, her son by a former marriage."
