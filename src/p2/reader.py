from os import listdir
from os.path import isfile, join
import json
import re
from snowballstemmer import TurkishStemmer
turkStem = TurkishStemmer()

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
    def read_all_data_and_get_crime_and_corpus(path='data'):
        jsonFileNames = [ str(i) + '.json' for i in range(1, 27842) ]
        crimeCorpusMap = {}
        for jsonFileName in jsonFileNames:
            content = Extractor.read_json_file(join(path, jsonFileName))
            crimeName = content['Suç']
            if crimeName == '':
                if 'undefined' not in crimeCorpusMap:
                    crimeCorpusMap['undefined'] = 0
                crimeName = 'undefined'
            if content['Suç'] not in crimeCorpusMap:
                crimeCorpusMap[content['Suç']] = 0
            # crimeCorpusMap[content['Suç']] = crimeCorpusMap[content['Suç']].append(content['ictihat'])
            crimeCorpusMap[crimeName] = crimeCorpusMap[crimeName] + 1
        return crimeCorpusMap
    
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



def stem_word(word):
    return turkStem.stemWord(word)