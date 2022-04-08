from genericpath import isfile
from ntpath import join
from os import listdir
from trnlp import TrnlpWord

tagger = TrnlpWord()

def tag_word(word):
    tagger.setword(word)
    return tagger.__str__()

def create_top_fifty(results_path):
    top_ten_f = open(results_path + "/top_fifty.log", "w")
    results_files = [result for result in listdir(results_path) if result.endswith(".txt")]
    for result_file in results_files:
        filename = result_file.split('.')[0]
        top_ten_f.write("[{}]\n".format(filename))
        with open("{}/{}".format(results_path, result_file), "r") as f:
            count = 0
            for line in f:
                word_group = line.split(",")
                original_words = word_group[0].split(" ")
                w1 = original_words[0]
                w2 = original_words[1]
                tagged_w1 = tag_word(w1)
                tagged_w2 = tag_word(w2)
                if count < 50 and len(w1) > 1 and len(w2) > 1 and "zarf" not in tagged_w1 and "zarf" not in tagged_w2 and "bağlaç" not in tagged_w1 and "bağlaç" not in tagged_w2 and not "GçDi" in tagged_w1 and not "GçDi" in tagged_w2 and "EfKŞrt" not in tagged_w1 and "EfKŞrt" not in tagged_w2 and "HeTyn" not in tagged_w1 and not "HeVas" in tagged_w1 and "HeTyn" not in tagged_w2 and not "HeVas" in tagged_w2 and "Gkz" not in tagged_w1 and "Gkz" not in tagged_w2 and "Fs" not in tagged_w2:
                    top_ten_f.write("{} {}\n".format(w1, w2))
                    count += 1
    top_ten_f.close()
