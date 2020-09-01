import argparse
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
from pyvi import ViTokenizer


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--text", default="Nỗi lòng thầm kín",
                    help="câu mở đầu lời thơ")
def main():
    args = parser.parse_args()
    tokenizer = Tokenizer()

    f = open("corpus.txt", "r")
    corpus = [sentence for sentence in f.read().lower().split("\n") if (sentence != "")]
    f.close()
    tokenizer.fit_on_texts(corpus)

    max_sequence_len = 56
    model = tf.keras.models.load_model('model/gen_poetry.h5')

    seed_text = args.text
    seed_text = ViTokenizer.tokenize(seed_text)
    print(seed_text)
    len_predict = len(seed_text.split())

    output_sentence = ""
    current = 0
    for i in range(len_predict):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0 )
        next_predict = np.array(model.predict(token_list)[0]).argsort()[-2:][::-1][-1]
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        if output_word == seed_text.split()[-1]:
            for word, index in tokenizer.word_index.items():
                if index == next_predict:
                    output_word = word
                    break
        if i == 0: seed_text += "\n"
        seed_text += " " + output_word

    # seed_text += "\n" + output_sentence
    print(seed_text)


if __name__ == "__main__":
    main()