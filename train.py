from tensorflow.keras.utils import get_file
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
import json, os
# variables
path = None
train_data = None
train_labels = None
test_data = None
test_labels = None
train_data_padded = None
test_data_padded = None
x_val = None
x_train = None
y_val = None
y_train = None
Model = None
fitModel = None
results = None
wordIndexFile = None
numpyArrayFile = None
########
#build model
def buildModel():
  embedding_dim = 16
  model = Sequential()
  model.add(Embedding(88000, embedding_dim))
  model.add(GlobalAveragePooling1D())
  model.add(Dense(16, activation="relu"))
  model.add(Dense(1, activation="sigmoid"))
  model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
  print(model.summary())
  return model
def loadDataForTraining(input):
  global wordIndexFile, numpyArrayFile
  fileName = input.rsplit('/', 1)[-1]
  file_path = get_file(fname =fileName,
                                    origin = input,
                                    extract=True)
  keras_datasets_dir = os.path.dirname(file_path)
  numpyArrayFile = os.path.join(keras_datasets_dir, \
                                searchFileInDir('imdb.npz',keras_datasets_dir))
  wordIndexFile = os.path.join(keras_datasets_dir,\
searchFileInDir('imdb_word_index.json',keras_datasets_dir))
def processDataForTraining():
  global train_data_padded, test_data_padded, x_train, x_test, y_train, y_test,x_val, x_training, y_val, y_training
  # variables
  start_char=1
  index_from=3
  oov_char=2
  skip_top=0
  seed=113
  num_words=10000
  with np.load(numpyArrayFile, allow_pickle=True) as f:
    x_train, labels_train = f["x_train"], f["y_train"]
    x_test, labels_test = f["x_test"], f["y_test"]
  rng = np.random.RandomState(seed)
  indices = np.arange(len(x_train))
  rng.shuffle(indices)
  x_train = x_train[indices]
  labels_train = labels_train[indices]
  indices = np.arange(len(x_test))
  rng.shuffle(indices)
  x_test = x_test[indices]
  labels_test = labels_test[indices]
  if start_char is not None:
    x_train = [[start_char] + [w + index_from for w in x] for x in x_train]
    x_test = [[start_char] + [w + index_from for w in x] for x in x_test]
  elif index_from:
    x_train = [[w + index_from for w in x] for x in x_train]
    x_test = [[w + index_from for w in x] for x in x_test]
  else:
    x_train = [[w for w in x] for x in x_train]
    x_test = [[w for w in x] for x in x_test]
  xs = x_train + x_test
  labels = np.concatenate([labels_train, labels_test])
  if not num_words:
    num_words = max(max(x) for x in xs)
  if oov_char is not None:
    xs = [
      [w if (skip_top <= w < num_words) else oov_char for w in x]
      for x in xs
    ]
  else:
    xs = [[w for w in x if skip_top <= w < num_words] for x in xs]
  idx = len(x_train)
  x_train, y_train = np.array(xs[:idx], dtype="object"), labels[:idx]
  x_test, y_test = np.array(xs[idx:], dtype="object"), labels[idx:]
  with open(wordIndexFile) as f:
    word_index = json.load(f)
    word_index = {k:(v+3) for k, v in word_index.items()}
    word_index['<PAD>'] = 0
    word_index['<START>'] = 1
    word_index['<UNK>'] = 2
    word_index['<UNUSED>'] = 3
  train_data_padded = pad_sequences(x_train, value=word_index['<PAD>'], padding ='post', maxlen = 250)
  test_data_padded = pad_sequences(x_test, value=word_index['<PAD>'], padding ='post', maxlen = 250)
  x_val = train_data_padded[:10000] # make first 10,000 data into validation
  x_training = train_data_padded[10000:] # make the data from 10,000 to end to be training
  # labels splitting
  y_val = y_train[:10000]
  y_training = y_train[10000:]
def trainTheModel():
  global fitModel
  fitModel = Model.fit(x_training, y_training, epochs=20, batch_size=512, validation_data=(x_val, y_val), verbose=1)
def evaluateModel():
  global results
  results = Model.evaluate(test_data_padded, y_test)
  print("Loss: ", results[0])
  print("Accuracy: ", results[1])
  # plot graphs to view loss and accuracy
  history_dict = fitModel.history
  history_dict.keys()
  acc = history_dict['accuracy']
  val_acc = history_dict['val_accuracy']
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']
  epochs = range(1, len(acc) + 1)
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()
  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(loc='lower right')
  plt.show()
def saveModel():
  Model.save("SentimentAnalysisModel.keras")
########helper function
def searchFileInDir(fileName,dir)->str:
  if fileName == "":
    return ""
  extension = fileName.split('.')[-1]
  files = [[f for f in os.listdir(dir) if f.endswith(type_)] for type_ in [extension]][0]
  for name in files:
    if name.find(fileName)>-1 :
      return name
Model = buildModel()
loadDataForTraining('https://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz')
processDataForTraining()
trainTheModel()
evaluateModel()
saveModel()
exit()