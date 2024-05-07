import nltk
from nltk.stem import WordNetLemmatizer

# Download the WordNet resource (if not already installed)
nltk.download('wordnet')

def to_singular(noun):
    lemmatizer = WordNetLemmatizer()
    singular_form = lemmatizer.lemmatize(noun, pos='n')  # 'n' denotes noun
    return singular_form


if __name__ == '__main__':
    print("Insert words to get thier singular form ")
    while(True):
        user_input = input("Please enter one noun word: ")
        if(user_input=="stop"):
            break
        print(to_singular(user_input))

