import numpy as np
from keras.models import model_from_json, load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import preprocess_input

from src.utils import load_pickle


# standard variables
MAX_LENGTH = 52
NPIX = 299
TARGET_SIZE = (NPIX,NPIX,3)

def load_model_from_json(path):
    with open(path,"r") as f:
        model = model_from_json(f.read())
    print("Model loaded successfully")
    return model

model = load_model_from_json("src/models/model.json")

# loading the weights of the model
model.load_weights("src/models/amey_19.h5")

# model = load_weights("src/models/model_final.h5", compile=False)
modified_inception = load_model("src/models/modified_inception.h5", compile=False)

index_and_words = load_pickle("src/models/index_and_words.pkl")
ix_to_word = index_and_words['ix_to_word']
word_to_ix = index_and_words['word_to_ix']


def greedy_search(photo):
    in_text = 'startseq'
    for i in range(MAX_LENGTH):
        sequence = [word_to_ix[w] for w in in_text.split() if w in word_to_ix]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ix_to_word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


def get_feature_vector(img_path):
    try:
        img = load_img(img_path, target_size=TARGET_SIZE)
    except OSError as e:
        print("Problem with image:",e)
        exit()
    
    # Converting image to array
    img_array = img_to_array(img)
    nimage = preprocess_input(img_array)
    
    # Adding one more dimension
    nimage = np.expand_dims(nimage, axis=0)    
    fea_vec = modified_inception.predict(nimage)
    return np.reshape(fea_vec, fea_vec.shape[1])


def generate_caption(img_path):
    image = get_feature_vector(img_path)
    image = image.reshape((1,2048))
    greedy = greedy_search(image)
    beam_k3 = beam_search_predictions(image)
    beam_k5 = beam_search_predictions(image, beam_index=5)
    return greedy, beam_k3, beam_k5

def beam_search_predictions(photo, beam_index=3):
    start = [word_to_ix["startseq"]]
    
    # start_word[0][0] = index of the starting word
    # start_word[0][1] = probability of the word predicted
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < MAX_LENGTH:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=MAX_LENGTH, padding='post')
            preds = model.predict([photo, par_caps], verbose=0)
            
            # Getting the top <beam_index>(n) predictions
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # creating a new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [ix_to_word[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption
