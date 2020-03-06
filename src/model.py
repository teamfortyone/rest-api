import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import preprocess_input

from src.utils import load_pickle


# standard variables
MAX_LENGTH = 52
NPIX = 299
TARGET_SIZE = (NPIX,NPIX,3)

model = load_model("src/models/coco_19th_epoch.h5", compile=False)
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
    
    # Adding one more dimesion
    nimage = np.expand_dims(nimage, axis=0)    
    fea_vec = modified_inception.predict(nimage)
    return np.reshape(fea_vec, fea_vec.shape[1])


def generateCaption(img_path):
    image = get_feature_vector(img_path)
    image = image.reshape((1,2048))
    caption = greedy_search(image)
    return caption
