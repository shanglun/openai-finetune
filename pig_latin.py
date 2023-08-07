import pandas as pd
import re

TRAINING_PASSAGE = '[YOUR PASSAGE HERE]'

def pig_latin(string):
    # if starts with a vowel, just add "ay"
    # else move the consonants to the end and add "ay"
    if string[0].lower() in {'a', 'e', 'i', 'o', 'u'}:
        return string + 'way'
    else:
        beginning_consonants = []
        for i in range(len(string)):
            if string[i].lower() in {'a', 'e', 'i', 'o', 'u'}:
                break
            beginning_consonants.append(string[i])
        return string[i:] + ''.join(beginning_consonants) + 'ay'

if __name__ == '__main__':
    toks = [t.lower() for t in re.split(r'\s', TRAINING_PASSAGE) if len(t) > 0]
    pig_latin_traindata = [
        {'prompt': 'Turn the following word into Pig Latin: %s \n\n###\n\n' % t, 'completion': '%s [DONE]' % pig_latin(t)}
        for t in toks
    ]
    pd.DataFrame(pig_latin_traindata).to_json('pig_latin.jsonl', orient='records', lines=True)
    # Run the training command here
    # export OPENAI_API_KEY=[OPENAI_API_KEY]
    # openai api fine_tunes.create -t pig_latin.jsonl -m davinci --suffix pig_latin

    model = '[YOUR MODEL NAME]'
    res = requests.post('https://api.openai.com/v1/completions', headers={
        'Content-Type': 'application/json',
        'Authorization': 'Bearer [OPENAI_API_KEY]'
    }, json={
        'prompt': "Turn the following word into Pig Latin: Latin \n\n###\n\n",
        'max_tokens': 100,
        'model': model,
        'stop': '[DONE]'
    })
    print(res.json()['choices'][0]['text']) # should print 'atinlay'

