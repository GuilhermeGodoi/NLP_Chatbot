import random
import json
import pickle
import numpy as np

import nltk
#nltk.download('punkt')       # ⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬
#nltk.download('wordnet')     # ENABLE IF NECESSARY
#nltk.download('omw-1.4')     # ⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫
from nltk.stem import WordNetLemmatizer     # Lemmatizer = Work --> Working, Worked, Works...

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD     #Stochastic gradient descent

print("Wait, do not quit!\nThis may take a few seconds")

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', "!", ".", ",","/","@",":", "//", "_", "-"]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) #"Tokenize" = "Hello im Guilherme" --> ['Hello', 'im', 'Guilherme']
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words)) #Eliminate duplicate words

classes = sorted(set(classes)) #Eliminate duplicate classes

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu')) #Input layer = dense layer with 256 neurons. The same goes for future codes (Not necessarily for input)
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #Stochastic gradient descent, lr = learning rate, don't be stuck to these values, just in what is passed
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=5, verbose=1) #epochs = feat same data 300 times in the NeuralNetwork
model.save('chatbot_model.h5', hist)
print("Done!")
