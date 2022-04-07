from extractor import Extractor
from chai_square import get_top_ten_for_chai_square
from gram_creator import get_word_map_for_payloads

def traverse_payloads_and_print_top_ten_for_chai_square(payloads):
    word_map, number_of_tokens_in_corpus = get_word_map_for_payloads(payloads)
    if number_of_tokens_in_corpus > 0:
        top_ten_for_chai_square = get_top_ten_for_chai_square(word_map, number_of_tokens_in_corpus)
        print("Top ten for chai square:")
        for i in range(len(top_ten_for_chai_square)):
                print("{}. {} {}".format(i + 1, top_ten_for_chai_square[i][0], top_ten_for_chai_square[i][1]))
        return
    print('No data for this day')

def main():
    days = list(range(20, 31))
    for current_day in days:
        print("Day {}".format(current_day))
        payloads = Extractor.read_data_by_day_and_get_payloads(day=str(current_day).zfill(2))
        traverse_payloads_and_print_top_ten_for_chai_square(payloads)

main()