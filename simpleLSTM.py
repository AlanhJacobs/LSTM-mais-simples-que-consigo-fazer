import pickle
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.optimizers import Adam

import matplotlib.pyplot as plt

perguntas = [] 
respostas = []

# abre o arquivo com as frases manuscritas
with open("dataset/perguntas.txt", 'r') as p:
    # le o arquivo
    perguntas = p.read()
    # separa em linhas
    perguntas = perguntas.split("\n")

with open("dataset/respostas.txt", 'r') as r:
    # le o arquivo
    respostas = r.read()
    # separa em linhas
    respostas = respostas.split("\n")


# cria um tokenizador
tokenizer = Tokenizer()
# pega o input e da um indice para cada palavra de acordo com a frequencia que ela aparece
tokenizer.fit_on_texts(perguntas + respostas)
# troca as palavras pelo seu indice
tokenized_perguntas = tokenizer.texts_to_sequences(perguntas)
tokenized_respostas = tokenizer.texts_to_sequences(respostas)
# obtenho o tamanho maximo de uma sequencia
max_sequence_length = max(max(len(seq) for seq in tokenized_perguntas), max(len(seq) for seq in tokenized_respostas))
# deixa todas as sequencias com o mesmo tamanho adicionando 0s na frente
padded_perguntas = pad_sequences(tokenized_perguntas, maxlen=max_sequence_length)
padded_respostas = pad_sequences(tokenized_respostas, maxlen=max_sequence_length)
# obtenho o tamanho do vocabulario
vocab_size = len(tokenizer.word_index) + 1

# cria o modelo
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_sequence_length))
# conjunto pequenos de dado, poucas camadas
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

optimizer = Adam(learning_rate=1e-2, weight_decay=1e-5)

# compila o modelo
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# treina o modelo
history  = model.fit(x=padded_perguntas, y=padded_respostas, epochs=200, verbose=2, batch_size=1)

model.save("simpleLSTM.keras")

# Salvando o tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# plotagem dos graficos
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.legend()
plt.show()