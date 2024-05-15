import logging
import os
import time
import json
from AnalyzerService import AnayzerSvc
import requests



def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


config = read_json("config.json")

baseApiUrl = config["baseApi"]
bucket_name = config["bucket_name"]
storeApi = baseApiUrl+config["storeApi"]
checkForFilesApi = baseApiUrl+config["checkForFilesApi"]

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config["google_creds"]


def readAudioFilesFromCloud(audioDetails):
    audios = []
    for audio in audioDetails:
        audioDet = {
            'Uri': audio['uri'],
            'Id': audio['id'],
            'IsAnalysisRequired': audio['isAnalysisRequired'],
            'LanguageCode': audio['languageCode'],
            'File': audio
        }
        audios.append(audioDet)

    return audios


class Analyzer:

    def __init__(self):
        self.isProcessDone = False
        self.analyzer = AnayzerSvc()


    def CheckIfAudioFilesExist(self):
        result = requests.get(checkForFilesApi, verify=False)
        jsonRes = result.json()
        anyExist = len(jsonRes['data']) > 0
        return anyExist, jsonRes['data']

    def ConvertAudioToText(self, FilesForTextConversion):
        Result = []
        for file in FilesForTextConversion:
            text = self.analyzer.transcribeFile(file['Path'], file['LanguageCode'])
            res = {
                'Id': file['Id'],
                'Text': text,
                'Sentiment': {'Score': 0, 'Result': "_"}
            }
            Result.append(res)
        return Result

    def AnalyzeAudio(self, ForAnalysis):
        Result = []
        for file in ForAnalysis:
            text = self.analyzer.transcribeFile(file['Uri'], file['LanguageCode'])
            sentiment = self.analyzer.analyzeText(text, file['LanguageCode'])
            res = {
                'Id': file['Id'],
                'Text': text,
                'Sentiment': sentiment
            }
            Result.append(res)
        return Result

    def process(self):
        isExist, audioFileDetails = self.CheckIfAudioFilesExist()
        if not isExist:
            #print('No Audio Files found to Process....Pausing Execution')
            return
        localAudioFiles = readAudioFilesFromCloud(audioFileDetails)

        print(str(len(localAudioFiles)) + ' Audio Files found to Process....Beginning Analysis')

        ForAnalysis = []
        TextConversion = []

        for file in localAudioFiles:
            if file['IsAnalysisRequired']:
                ForAnalysis.append(file)
            else:
                TextConversion.append(file)

        TranscribeResult = self.ConvertAudioToText(TextConversion)
        AnalysisResult = self.AnalyzeAudio(ForAnalysis)

        AnalysisResult = TranscribeResult + AnalysisResult

        # send these results to .Net controller and make changes to db from there

        # verify must be false only for localhost calls as done in server
        requests.post(storeApi, json=AnalysisResult, verify=False)
        self.isProcessDone = True


if __name__ == '__main__':

    while True:
        analyzer = Analyzer()
        logging.log(level=logging.ERROR, msg='Beginning Execution...')
        analyzer.process()
        time.sleep(10)
