from google.cloud import speech
from google.cloud import language_v1

from SentimentAnalyzer import SentimentAnalyzer


def formatSentimentResult(score):

    if score > 0.25:
        senti = 'Positive'
        score = score * 100

    elif score < -0.25:
        senti = 'Negative'
        score = score * 100 * -1
    else:
        senti = 'Neutral'
        score = (score + 0.25) * 200

    res = dict(
        Score=score,
        Result=senti
    )
    return res

class AnayzerSvc:

    def transcribeFile(self, Uri, languageCode):

        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(uri=Uri)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
            language_code=languageCode,
        )

        operation = client.long_running_recognize(config=config, audio=audio)

        #print("Waiting for operation to complete...")
        response = operation.result(timeout=90)

        # Each result is for a consecutive portion of the audio. Iterate through
        # them to get the transcripts for the entire audio file.

        finalText = str()
        for result in response.results:
            finalText = finalText + result.alternatives[0].transcript
        return finalText

    def analyzeText(self, text, languageCode):
        if languageCode == 'en-US':
            analyzer = SentimentAnalyzer()
            conf, sentiment = analyzer.predict_sentiment(text)
            res = dict(
                Score=conf.item(),
                Result=sentiment
            )
            return res

        # init google language client
        client = language_v1.LanguageServiceClient()
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)

        # Detects the sentiment of the text
        sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment

        return formatSentimentResult(sentiment.score)

