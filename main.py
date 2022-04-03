import json


f = open ('./1.json', "r")
 
data = json.loads(f.read())
payload = data['ictihat'].strip().split(' ') # We should remove spaces in the text

wordMap = {}
gramCount = 5
indicator = gramCount // 2


for i in range(indicator, len(payload) - indicator):
    word = payload[i]
    if word not in wordMap:
        wordMap[word] = {
            -2: {},
            -1: {},
            0: {},
            1: {},
            2: {},
        }
    frequencyMap = wordMap[word]
    for j in range(gramCount):
        gram = payload[i - indicator + j]
        if gram in frequencyMap[-indicator + j]:
            frequencyMap[-indicator + j][gram] += 1
        else:
            frequencyMap[-indicator + j][gram] = 1

ve = wordMap['ve']

for i in range(-2, 2):
    for gram in ve[i]:
        print(i, gram, ve[i][gram])