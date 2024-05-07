from GrocerySentencesClassifier import GrocerySentenceEmbeddingClassifier

import torch
import requests
import os

# #path = 'C:\\Users\\user1\\PycharmProjects\\DsShoppinglistModels'
# #Load the model from disk
# #model = SentenceEmbeddingClassifier.load_from_checkpoint(checkpoint_path="C:\\Users\\user1\\PycharmProjects\\DsShoppinglist\\BestModels\\epoch=4-val_acc=0.99.ckpt")
# model = SentenceEmbeddingClassifier('mixedbread-ai/mxbai-embed-large-v1', total_steps=10)
# # model.load_state_dict(torch.load(path+'\\grocery-sentence-classifier.pth'))
# model_url = 'https://huggingface.co/arpnir/DsShoppinglist/resolve/main/grocery-sentence-classifier.pth'
# #model_weights = torch.hub.load_state_dict_from_url(model_url, map_location=torch.device('cpu'))
# # Download the file and save it temporarily
# print("downloading model from hugging face...")
# r = requests.get(model_url)
# with open('temp_model_weights.pth', 'wb') as f:
#     f.write(r.content)
# print("download complited.")
# model_weights = torch.load('temp_model_weights.pth', map_location='cpu')
# model.load_state_dict(model_weights)
# model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)


model = None
def init_grocery_model():
    global model
    model = GrocerySentenceEmbeddingClassifier('mixedbread-ai/mxbai-embed-large-v1', total_steps=10)
    model_weights_file_path = 'temp_model_weights.pth'

    if not os.path.exists(model_weights_file_path):
        model_url = 'https://huggingface.co/arpnir/DsShoppinglist/resolve/main/grocery-sentence-classifier.pth'
        print("downloading model from hugging face...")
        r = requests.get(model_url)
        with open('temp_model_weights.pth', 'wb') as f:
            f.write(r.content)
        print("download complited.")

    model_weights = torch.load('temp_model_weights.pth', map_location='cpu')
    model.load_state_dict(model_weights)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

def is_grocery_sentence(input):
    #Make the inferance
    with torch.no_grad():
        output = model((input,))
        # If your model includes a softmax layer for classification at the end
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        if (probabilities[1]>0.9):
            return True
        else:
            return False

if __name__ == '__main__':
    path = 'C:\\Users\\user1\\PycharmProjects\\DsShoppinglistModels'
    while(True):
        user_input = input("Please enter something: ")

        if(user_input=="stop"):
            break

        if(user_input=="save"):
            print("Saving the model and its config file...")
            # Assuming 'model' is your LightningModule instance

            torch.save(model.state_dict(), path+'\\grocery-sentence-classifier.pth')
            # Open the file for writing ('w' mode)
            with open(path+'\\config.json', 'w') as file:
                orig_model = GrocerySentenceEmbeddingClassifier('mixedbread-ai/mxbai-embed-large-v1', total_steps=10)
                file.write(orig_model.get_configoration())
            print("Model and config file saved.")
            continue

        grocery = is_grocery_sentence(user_input)
        if(grocery):
            print("This is a grocery sentence")
        else:
            print("This is NOT a grocery sentence")