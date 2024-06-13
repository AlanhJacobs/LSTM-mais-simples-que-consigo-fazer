import keras

from keras.layers import LSTM, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential

input = []

with open("frases_manuscritas.txt", 'r') as file:
    input = file.read()
    input = input.split("\n")

tokenizer = Tokenizer()
# pega o input e da um indice para cada palavra de acordo com a frequencia que ela aparece
tokenizer.fit_on_texts(input)
# troca as palavras pelo seu indice
sequences = tokenizer.texts_to_sequences(input)
# deixa todas as sequencias com o mesmo tamanho adicionando 0s
padded_sequences = pad_sequences(sequences)

print(padded_sequences)

model = Sequential()
model.add(Embedding(input_dim=len))