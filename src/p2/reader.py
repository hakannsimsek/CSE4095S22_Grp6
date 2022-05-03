from os import listdir
from os.path import isfile, join
import json
import re
from snowballstemmer import TurkishStemmer
turkStem = TurkishStemmer()

class Reader:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
          cls.instance = super(Reader, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def clear_punctuations(payload):
        return re.sub(r'[^\w\s]' ,'' ,payload)
    
    @staticmethod
    def read_all_data_and_get_payloads(path='data'):
        jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
        payloads = []
        for jsonFileName in jsonFileNames:
            payloads.append(Reader.read_json_file_and_get_payload(join(path, jsonFileName)))
        return payloads
    
    @staticmethod
    def read_all_data_and_get_crime_map_and_docs_and_doc_crime_list(path='data'):
        jsonFileNames = [ str(i) + '.json' for i in range(1, 27000) ]
        crimeCorpusMap = {}
        doc_crime_list = []
        docs = []
        for jsonFileName in jsonFileNames:
            content = Reader.read_json_file(join(path, jsonFileName))
            crimeName = content['Suç']
            if crimeName == '':
                if 'undefined' not in crimeCorpusMap:
                    crimeCorpusMap['undefined'] = { 'corpus': [], 'count': 0 }
                crimeName = 'undefined'
            if crimeName not in crimeCorpusMap:
                crimeCorpusMap[content['Suç']] = { 'corpus': [], 'count': 0 }
            doc_crime_list.append(crimeName)
            crimeCorpusMap[crimeName]['corpus'].append(content['ictihat'])
            crimeCorpusMap[crimeName]
            docs.append(content['ictihat'])
            crimeCorpusMap[crimeName]['count'] = crimeCorpusMap[crimeName]['count'] + 1
        return crimeCorpusMap, docs, doc_crime_list
    
    @staticmethod
    def read_some_data_and_get_payloads(path='data', number_of_files=100):
        jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
        payloads = []
        for jsonFileName in jsonFileNames[:number_of_files]:
            payloads.append(Reader.read_json_file_and_get_payload(join(path, jsonFileName)))
        return payloads

    @staticmethod
    def read_data_by_day_and_get_payloads(path='data', day='04'):
        jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
        payloads = []
        for jsonFileName in jsonFileNames:
            data = Reader.read_json_file(join(path, jsonFileName))
            if data['Mahkeme Günü'] == day:
                payloads.append(Reader.get_payload(data))
        return payloads

    @staticmethod
    def read_data_by_day_and_get_payloads_day_map(path='data'):
        jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
        payloads_day_map = {}
        for jsonFileName in jsonFileNames:
            data = Reader.read_json_file(join(path, jsonFileName))
            day = data['Mahkeme Günü']
            payloads_day_map[day] = payloads_day_map.get(day, []) + [Reader.get_payload(data)]
        return payloads_day_map

    @staticmethod
    def read_json_file(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    @staticmethod
    def get_payload(data):
        payload = Reader.clear_punctuations(data['ictihat'].lower().strip()).split(' ')
        return [ele for ele in payload if ele.strip()]

    @staticmethod
    def read_json_file_and_get_payload(file_name):
        return Reader.get_payload(Reader.read_json_file(file_name))

    @staticmethod
    def getPlainTextFromPayloads(payloads):
        flatPayload = [item for sublist in payloads for item in sublist]
        return ' '.join(flatPayload)



def stem_word(word):
    return turkStem.stemWord(word)