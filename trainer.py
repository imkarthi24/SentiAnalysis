import logging
import os
import time
import json
from SentimentAnalyzer import SentimentAnalyzer
import requests


if __name__ == '__main__':

    while True:
        analyzer = SentimentAnalyzer()
        text = "Worst movie ever"
        analyzer.predict_sentiment(text)
