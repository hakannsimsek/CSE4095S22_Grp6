word_map = {}

def get_word_map(payload):
    gram_count = 5
    indicator = gram_count // 2

    for i in range(indicator, len(payload) - indicator):
        word = payload[i]
        if word.isalpha() and word != "":
            if word not in word_map:
                word_map[word] = {
                    -2: {},
                    -1: {},
                    0: {},
                    1: {},
                    2: {},
                    'count': 0
                }
            frequency_map = word_map[word]
            frequency_map['count'] += 1
            for j in range(gram_count):
                gram = payload[i - indicator + j]
                if gram in frequency_map[-indicator + j]:
                    frequency_map[-indicator + j][gram] += 1
                else:
                    frequency_map[-indicator + j][gram] = 1
    return len(payload)

def get_word_map_for_payloads(payloads):
    number_of_tokens_in_corpus_sum = 0

    for payload in payloads:
        number_of_tokens_in_corpus_sum += get_word_map(payload)

    return word_map, number_of_tokens_in_corpus_sum