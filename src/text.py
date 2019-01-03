import os
import re
import string


path_data_neg = './data/NEG'
path_data_pos = './data/POS'
path_data_tag_neg = './data-tagged/NEG'
path_data_tag_pos = './data-tagged/POS'

punctuations = string.punctuation+'``'+'"'

def read_data_from_file(sentiment):
    # para sentiment: whether the review is neg or pos
    # return para: a list of words along without their tags
    # return type: list(list(str))
    path = path_data_neg if sentiment == 'neg' else path_data_pos
    files = os.listdir(path)
    reviews = []
    for file in files:
        if not os.path.isdir(file):
            f = open(path+'/'+file, 'r', encoding='utf-8')
            review = list()
            for line in f:
                for word in re.split('\W+', line):
                    review.append(word)
            reviews.append(review)
            f.close()
    return reviews


def read_data_tag_from_file(sentiment):
    # para sentiment: whether the review is neg or pos
    # return para: a list of words along with their tags
    # return type: list(list(tuple(str,str)))
    path = path_data_tag_neg if sentiment == 'neg' else path_data_tag_pos
    files = os.listdir(path)
    reviews_tags = []
    for file in files:
        if not os.path.isdir(file):
            f = open(path+'/'+file, 'r', encoding='utf-8')
            review_tag = list()
            for line in f:
                word_tag = re.split(r'\t+', line[:-1])
                if len(word_tag) == 2 and word_tag[0] not in punctuations:
                    review_tag.append((word_tag[0],word_tag[1]))
            reviews_tags.append(review_tag)
            f.close()  # otherwise resource warning
    return reviews_tags


def visual_data(reviews):
    # para tags: a list of words 
    # just visualise tags for one document
    print("--"*20)
    for word in reviews[0]:
        print(word)


def visual_data_tag(tags):
    # para tags: a list of words along with their tags
    # just visualise tags for one document
    print("--"*20)
    print("word \t\t count")
    for (word, tag) in tags[0]:
        print(word.ljust(20), tag)
