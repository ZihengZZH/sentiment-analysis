import os
import re


path_data_tag_neg = './data-tagged/NEG'
path_data_tag_pos = './data-tagged/POS'


def read_data_tag_from_file(sentiment):
    # para sentiment: whether the review is neg or pos
    # return para: a list of words along with their tags
    # return type: list(list(tuple(str,str)))
    path = path_data_tag_neg if sentiment == 'neg' else path_data_tag_pos
    files = os.listdir(path)
    tags = []
    for file in files:
        if not os.path.isdir(file):
            f = open(path+'/'+file, 'r', encoding='utf-8')
            tag = list()
            for line in f:
                word_tag = re.split(r'\t+', line[:-1])
                if len(word_tag) == 2:
                    tag.append((word_tag[0],word_tag[1]))
            tags.append(tag)
            f.close()  # otherwise resource warning
    return tags


def visual_data_tag(tags):
    # para tags: a list of words along with their tags
    # just visualise tags for one document
    print("--"*20)
    print("word \t\t count")
    for (word, tag) in tags[0]:
        print(word.ljust(20), tag)
