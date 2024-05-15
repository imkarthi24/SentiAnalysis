import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow
import pandas as pd


class SentimentAnalyzer:
    def __init__(self):
        modelFilePath = os.path.join(os.getcwd(), 'sentiment_model.h5')
        datasetPath = os.path.join(os.getcwd(), 'IMDBDataset.csv')
        df = pd.read_csv(datasetPath)

        reviewDf = df[['review', 'sentiment']]
        self.sentiment_label = reviewDf.sentiment.factorize()

        review = reviewDf.review.values

        self.tokenizer = Tokenizer(num_words=5000)
        self.tokenizer.fit_on_texts(review)
        vocab_size = len(self.tokenizer.word_index) + 1
        encoded_docs = self.tokenizer.texts_to_sequences(review)
        padded_sequence = pad_sequences(encoded_docs, maxlen=200)

        self.model = tensorflow.keras.models.load_model(modelFilePath)

    # returns Confidence and Sentiment
    def predict_sentiment(self, text):
        tw = self.tokenizer.texts_to_sequences([text])
        tw = pad_sequences(tw, maxlen=200)
        prediction = self.model.predict(tw).argmax()
        score = max(self.model.predict(tw)[0])
        return score * 100, self.sentiment_label[1][prediction]
