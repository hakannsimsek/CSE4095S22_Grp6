import scipy.stats

critical_chai_square_value = scipy.stats.chi2.ppf(1-.05, df=1)

def calculate_chai_square(word_frequency_matrix, number_of_tokens_in_corpus):
    return number_of_tokens_in_corpus * pow((word_frequency_matrix[0][0] * word_frequency_matrix[1][1]) - (word_frequency_matrix[0][1] * word_frequency_matrix[1][0]), 2) / ((word_frequency_matrix[0][0] + word_frequency_matrix[1][0]) * (word_frequency_matrix[0][1] + word_frequency_matrix[1][1]) * (word_frequency_matrix[0][0] + word_frequency_matrix[0][1]) * (word_frequency_matrix[1][0] + word_frequency_matrix[1][1]))

def construct_frequency_matrix_from_word_map(w1, w2, word_map, number_of_tokens_in_corpus):
    # [[w1w2, w1!w2], [w1w2!, w1!w2!]]
    compound_word_count = 0
    if w2 in word_map[w1][1]:
        compound_word_count = word_map[w1][1][w2]
    w1_count = word_map[w1]['count']
    w2_count = word_map[w2]['count']
    frequency_matrix = [[compound_word_count, w2_count - compound_word_count],[w1_count - compound_word_count,number_of_tokens_in_corpus + 2 * compound_word_count - w1_count - w2_count]]
    return frequency_matrix

def get_top_ten_for_chai_square(word_map, number_of_tokens_in_corpus):
    chai_square_list = []
    for w1 in word_map:
        for w2 in word_map:
            if w1 != w2:
                frequency_matrix = construct_frequency_matrix_from_word_map(w1, w2, word_map, number_of_tokens_in_corpus)
                chai_square_value = calculate_chai_square(frequency_matrix, number_of_tokens_in_corpus)
                if chai_square_value != number_of_tokens_in_corpus and chai_square_value > critical_chai_square_value:
                    chai_square_list.append([w1, w2, chai_square_value])
    chai_square_list.sort(key=lambda x: x[2], reverse=True)
    return chai_square_list[:10]