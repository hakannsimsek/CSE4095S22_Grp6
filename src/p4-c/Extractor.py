from os import listdir
from os.path import isfile, join
import json
import re
from random import randint

from snowballstemmer import TurkishStemmer

turkStem = TurkishStemmer()


class Extractor:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Extractor, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def clear_punctuations(payload):
        return re.sub(r'[^\w\s]', '', payload)

    @staticmethod
    def read_all_data_and_get_payloads(path='data'):
        jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
        payloads = []
        for jsonFileName in jsonFileNames:
            payloads.append(Extractor.read_json_file_and_get_payload(join(path, jsonFileName)))
        return payloads

    @staticmethod
    def read_some_data_and_get_payloads_and_crimes(path='data', number_of_files=100):
        # jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
        crime_repetition_list = Extractor.get_crime_repetition_list(path=path, number_of_files=number_of_files)
        jsonFileNames = [str(i) + '.json' for i in range(1, number_of_files)]
        payloads = []
        crimes = []
        for jsonFileName in jsonFileNames[:number_of_files]:
            payload, crime = Extractor.read_json_file_and_get_payload_and_crime(join(path, jsonFileName))
            if crime_repetition_list[crime] > 5:
                payloads.append(payload)
                crimes.append(crime)
        return payloads, crimes

    @staticmethod
    def read_some_data_and_get_crimes(path='data', number_of_files=100):
        # jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
        jsonFileNames = [str(i) + '.json' for i in range(1, number_of_files)]
        crimes = []
        for jsonFileName in jsonFileNames[:number_of_files]:
            crimes.append(Extractor.read_json_file_and_get_crime(join(path, jsonFileName)))
        return crimes

    @staticmethod
    def get_crime_repetition_list(path='data', number_of_files=100):
        jsonFileNames = [str(i) + '.json' for i in range(1, number_of_files)]
        crime_repetition_map = {}
        for jsonFileName in jsonFileNames[:number_of_files]:
            crime = Extractor.read_json_file_and_get_crime(join(path, jsonFileName))
            if crime in crime_repetition_map:
                crime_repetition_map[crime] = crime_repetition_map[crime] + 1
            else:
                crime_repetition_map[crime] = 1
        return crime_repetition_map

    @staticmethod
    def read_data_by_day_and_get_payloads(path='data', day='04'):
        jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
        payloads = []
        for jsonFileName in jsonFileNames:
            data = Extractor.read_json_file(join(path, jsonFileName))
            if data['Mahkeme Günü'] == day:
                payloads.append(Extractor.get_payload(data))
        return payloads

    @staticmethod
    def read_data_by_day_and_get_payloads_day_map(path='data'):
        jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
        payloads_day_map = {}
        for jsonFileName in jsonFileNames:
            data = Extractor.read_json_file(join(path, jsonFileName))
            day = data['Mahkeme Günü']
            payloads_day_map[day] = payloads_day_map.get(day, []) + [Extractor.get_payload(data)]
        return payloads_day_map

    @staticmethod
    def read_json_file(filename):
        with open(filename, 'r', encoding="utf-8") as f:
            data = json.load(f)
        return data

    @staticmethod
    def get_payload(data):
        payload = Extractor.clear_punctuations(data['ictihat'].lower().strip()).split(' ')
        return [ele for ele in payload if ele.strip()]

    @staticmethod
    def get_payloadd(data):
        payload = (data['ictihat'].lower())
        return payload

    @staticmethod
    def get_crime(data):
        payload = ""
        if data['Suç'] != "":
            payload = (data['Suç'].lower().strip()).split(',')[0]
        return payload

    @staticmethod
    def read_json_file_and_get_payload_and_crime(file_name):
        content = Extractor.read_json_file(file_name)
        return Extractor.get_payloadd(content), Extractor.get_crime(content)

    @staticmethod
    def read_json_file_and_get_crime(file_name):
        return Extractor.get_crime(Extractor.read_json_file(file_name))

    @staticmethod
    def getPlainTextFromPayloads(payloads):
        flatPayload = [item for sublist in payloads for item in sublist]
        return ' '.join(flatPayload)


def stem_word(word):
    return turkStem.stemWord(word)