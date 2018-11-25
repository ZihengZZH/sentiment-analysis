import os

path_data_neg = './data/NEG'
path_data_pos = './data/POS'
path_tag_neg = './data-tagged/NEG'
path_tag_pos = './data-tagged/POS'

def read_data_from_file(sentiment):
    # para sentiment: whether the review is neg or pos
    path = path_data_neg if sentiment == 'neg' else path_data_pos
    files = os.listdir(path)
    texts = []
    for file in files:
        if not os.path.isdir(file):
            f = open(path+'/'+file, 'r', encoding='utf-8')
            iter_f = iter(f)
            review = ""
            for line in iter_f:
                review += line
            texts.append(review)
            f.close() # otherwise resource warning
    return texts

def read_tag_from_file(sentiment):
    # para sentiment: whether the review is neg or pos
    # return a list of tags in which each word has been tagged 
    path = path_tag_neg if sentiment == 'neg' else path_tag_pos
    files = os.listdir(path)
    tags = []
    for file in files:
        if not os.path.isdir(file):
            f = open(path+'/'+file, 'r', encoding='utf-8')
            iter_f = iter(f)
            tag = ""
            for line in iter_f:
                tag += line
            tags.append(tag)
            f.close()
    return tags