from extractor import Extractor
from chai_square import get_top_thousand_for_chai_square
from gram_creator import get_word_map_for_payloads
import os
from extractor import stem_word
from tagger import create_top_fifty
import pmi

def traverse_payloads_and_print_top_thousand_for_chai_square(day, payloads):
    word_map, number_of_tokens_in_corpus = get_word_map_for_payloads(payloads)
    if number_of_tokens_in_corpus > 0:
        top_thousand_for_chai_square = get_top_thousand_for_chai_square(word_map, number_of_tokens_in_corpus) # [w1, w2, value]
        # Create directory if it doesn't exist
        raw_filename = "chai_square_results/{}.txt".format(day)
        try:
            os.remove(raw_filename)
        except:
            pass
        raw_f = open(raw_filename, 'a')
        print("Top ten for chai square:")
        for i in range(len(top_thousand_for_chai_square)):
            w1, w2 = top_thousand_for_chai_square[i][0], top_thousand_for_chai_square[i][1]
            raw_f.write("{} {},{} {}\n".format(w1, w2, stem_word(w1), stem_word(w2)))
        print("Day {} is written in {}".format(day, raw_filename))
        raw_f.close()
        return
    print('No data for this day')

def main():
    if not os.path.exists(os.path.dirname("/chai_square_results")):
        os.makedirs(os.path.dirname("/chai_square_results"))
    for day in range(20, 31):
        payloads = Extractor.read_data_by_day_and_get_payloads(day=str(day).zfill(2))
        print("Number of payloads: {}".format(len(payloads)))
        print("Day {}".format(day))
        traverse_payloads_and_print_top_thousand_for_chai_square(day, payloads)
    create_top_fifty("chai_square_results")
    payloads = Extractor.read_some_data_and_get_payloads()
    # word_map, number_of_tokens_in_corpus = get_word_map_for_payloads(payloads)
    # print_pmi(word_map, number_of_tokens_in_corpus)

main()