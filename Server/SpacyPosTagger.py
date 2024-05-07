import spacy

# Load the English NLP model
#Run this line first in your venv : 'python -m spacy download en_core_web_sm'
nlp = spacy.load("en_core_web_sm")

def tag_tokens(text):
    # Process the text
    doc = nlp(text)
    taged_tokens = []
    # Display the words and their POS tags
    for token in doc:
        taged_tokens.append((token.text, token.pos_))
    return taged_tokens

def has_noun(text):
    tagged_tokens = tag_tokens(text)
    nouns = [word for word, pos in tagged_tokens if pos in ['NOUN']]
    has_nouns = len(nouns) > 0
    return has_nouns

def has_noun_or_adj(text):
    tagged_tokens = tag_tokens(text)
    nouns = [word for word, pos in tagged_tokens if pos in ['NOUN','ADJ']]
    has_nouns = len(nouns) > 0
    return has_nouns

if __name__ == '__main__':

    while(True):
        user_input = input("Please enter some text: ")
        if(user_input=="stop"):
            break
        print(tag_tokens(user_input))
        print("has_noun: " + str(has_noun(user_input)))