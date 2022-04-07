from os import listdir
from os.path import isfile, join
import json
import re

class Extractor:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
          cls.instance = super(Extractor, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def clear_punctuations(payload):
        return re.sub(r'[^\w\s]' ,'' ,payload)
    
    @staticmethod
    def read_all_data_and_get_payloads(path='data'):
        jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
        payloads = []
        for jsonFileName in jsonFileNames:
            payloads.append(Extractor.read_json_file_and_get_payload(join(path, jsonFileName)))
        return payloads
    
    @staticmethod
    def read_some_data_and_get_payloads(path='data', number_of_files=100):
        jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
        payloads = []
        for jsonFileName in jsonFileNames[:number_of_files]:
            payloads.append(Extractor.read_json_file_and_get_payload(join(path, jsonFileName)))
        return payloads

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
    def read_json_file(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    @staticmethod
    def get_payload(data):
        payload = Extractor.clear_punctuations(data['ictihat'].lower().strip()).split(' ')
        return [ele for ele in payload if ele.strip()]

    @staticmethod
    def read_json_file_and_get_payload(file_name):
        return Extractor.get_payload(Extractor.read_json_file(file_name))

    @staticmethod
    def getPlainTextFromPayloads(payloads):
        flatPayload = [item for sublist in payloads for item in sublist]
        return ' '.join(flatPayload)