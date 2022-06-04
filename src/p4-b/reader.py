from os import listdir
from os.path import isfile, join
import json
import re
from snowballstemmer import TurkishStemmer
turkStem = TurkishStemmer()

number_of_docs = 27842

def filterFunc(courtName):
    def _innerFilter(word):
        return word not in courtName

    return _innerFilter

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
    def split_crime_then_fetch_first_one(crime):
        return crime.split(',')[0]

    @staticmethod
    def read_all_data_and_get_crime_and_corpus(path='data'):
        jsonFileNames = [str(i) + '.json' for i in range(1, 1000)]
        crimeCorpusMap = {}
        crimeList = []
        for jsonFileName in jsonFileNames:
            name = join(path, jsonFileName)
            content = Reader.read_json_file(name)
            crimeList.append(content)
            crimeName = content['Suç']
            if crimeName == '':
                if 'undefined' not in crimeCorpusMap:
                    crimeCorpusMap['undefined'] = 0
                crimeName = 'undefined'
            if content['Suç'] not in crimeCorpusMap:
                crimeCorpusMap[content['Suç']] = 0
            # crimeCorpusMap[content['Suç']] = crimeCorpusMap[content['Suç']].append(content['ictihat'])
            crimeCorpusMap[crimeName] = crimeCorpusMap[crimeName] + 1
        return crimeCorpusMap, 
    
    @staticmethod
    def read_all_data_and_get_crime_map_and_docs_and_doc_crime_list(path='data'):
        jsonFileNames = [ str(i) + '.json' for i in range(1, 10000) ]
        crimeCorpusMap = {}
        doc_crime_list = []
        docs = []
        for jsonFileName in jsonFileNames:
            content = Reader.read_json_file(join(path, jsonFileName))
            crimeName = Reader.split_crime_then_fetch_first_one(content['Suç'])
            if crimeName == '':
                if 'undefined' not in crimeCorpusMap:
                    crimeCorpusMap['undefined'] = { 'corpus': [], 'count': 0 }
                crimeName = 'undefined'
            if crimeName not in crimeCorpusMap:
                crimeCorpusMap[crimeName] = { 'corpus': [], 'count': 0 }
            doc_crime_list.append(crimeName)
            crimeCorpusMap[crimeName]['corpus'].append(content['ictihat'])
            crimeCorpusMap[crimeName]
            docs.append(content['ictihat'])
            crimeCorpusMap[crimeName]['count'] = crimeCorpusMap[crimeName]['count'] + 1
        return crimeCorpusMap, docs, doc_crime_list

    @staticmethod
    def read_all_data_and_get_court_map_and_docs_and_doc_court_list(path='data'):
<<<<<<< HEAD
        jsonFileNames = [ str(i) + '.json' for i in range(1, number_of_docs) ]
=======
        jsonFileNames = [ str(i) + '.json' for i in range(1, 100) ]
>>>>>>> 2e8e9e6b1ef2f537b9fa566ee3f5aec02dfb1766
        court_corpus_map = {
            "Asliye Ceza Mahkemesi": { 'corpus': [], 'count': 0 },
            "Ağır Ceza Mahkemesi": { 'corpus': [], 'count': 0 },
            "EMPTY": { 'corpus': [], 'count': 0 },
            "Asliye Hukuk Mahkemesi": { 'corpus': [], 'count': 0 },
            "Ceza Dairesi": { 'corpus': [], 'count': 0 },
            "İş Mahkemesi": { 'corpus': [], 'count': 0 },
            "Bölge Adliye Mahkemesi": { 'corpus': [], 'count': 0 },
            "Çocuk Mahkemesi": { 'corpus': [], 'count': 0 },
            "Ticaret Mahkemesi": { 'corpus': [], 'count': 0 },
            "OTHER": { 'corpus': [], 'count': 0 },
        }
        doc_court_list = []
        docs = []
        for jsonFileName in jsonFileNames:
            content = Reader.read_json_file(join(path, jsonFileName))
            courtName = content['Mahkemesi'].strip()
            updated_court_name = courtName
            is_not_in_defined_courts = True
            for defined_court in court_corpus_map.keys():
                if defined_court in courtName:
                    updated_court_name = defined_court
                    is_not_in_defined_courts = False
                    break
            if courtName == '':
                court_corpus_map['EMPTY']['corpus'].append(content['ictihat'])
                court_corpus_map['EMPTY']['count'] = court_corpus_map['EMPTY']['count'] + 1
                updated_court_name = 'EMPTY'
            elif is_not_in_defined_courts:
                court_corpus_map['OTHER']['corpus'].append(content['ictihat'])
                court_corpus_map['OTHER']['count'] = court_corpus_map['OTHER']['count'] + 1
                updated_court_name = 'OTHER'
            else:
                original_court_name = courtName
                for defined_court in court_corpus_map.keys():
                    if defined_court in courtName:
                        original_court_name = defined_court
                updated_court_name = original_court_name
                # for court_name in court_corpus_map.keys():
                #     if court_name in courtName:
                #         original_court_name = court_name
                #         break
                if original_court_name == 'Çocuk Ağır Ceza Mahkemesi':
                    print('Boom')
                court_corpus_map[original_court_name]['corpus'].append(content['ictihat'])
                court_corpus_map[original_court_name]['count'] = court_corpus_map[original_court_name]['count'] + 1
            doc_court_list.append(updated_court_name)
            docs.append(content['ictihat'])
        return court_corpus_map, docs, doc_court_list
    
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