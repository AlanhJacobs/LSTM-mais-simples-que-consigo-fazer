import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model("simpleLSTM.keras")
while True:
    # Exemplo de nova pergunta para teste
    nova_pergunta = input()

    # Tokenizar a nova pergunta
    tokenized_nova_pergunta = tokenizer.texts_to_sequences([nova_pergunta])

    # Padronizar a sequência da nova pergunta
    padded_nova_pergunta = pad_sequences(tokenized_nova_pergunta, maxlen=6)

    # Realizar a previsão com o modelo
    predicted_sequence = model.predict(padded_nova_pergunta)

    # A saída do modelo é uma sequência de probabilidades para cada palavra no vocabulário
    # Você pode decodificar isso para obter a resposta predita
    # Aqui está um exemplo básico de decodificação
    predicted_index = np.argmax(predicted_sequence, axis=-1)[0]

    # Decodificar o índice predito de volta para palavras
    predicted_answer = tokenizer.sequences_to_texts([predicted_index])[0]

    print(f"Pergunta: {nova_pergunta}")
    print(f"Resposta predita: {predicted_answer}")