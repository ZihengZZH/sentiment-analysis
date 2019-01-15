import os
import re
import string
import glob
import requests
import tarfile
import sys
import codecs
from smart_open import smart_open


path_data_neg = './dataset/data/NEG'
path_data_pos = './dataset/data/POS'
path_data_tag_neg = './dataset/data-tagged/NEG'
path_data_tag_pos = './dataset/data-tagged/POS'

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
                    if word != '':
                        review.append(word.lower())
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
                    review_tag.append((word_tag[0].lower(),word_tag[1]))
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


'''ONLY EXECUTED ONCE'''
# prepare the IMDB data (normalization and cleaning)
def prepare_data_IMDB():
    dirname = 'aclImdb'
    filename = './dataset/aclImdb_v1.tar.gz'
    all_lines = []
    control_chars = [chr(0x85)] # Py3

    # convert text to lower-case and strip punctuation/symbols from words
    def normalize_text(text):
        norm_text = text.lower()
        # replace breaks with spaces
        norm_text = norm_text.replace('<br />', ' ')
        # pad punctuation with spaces on both sides
        norm_text = re.sub(r"([\.\",\(\)!\?;:])", " \\1", norm_text)
        return norm_text

    if not os.path.isfile("./dataset/aclImdb/alldata-id.txt"):
        if not os.path.isdir(dirname):
            tar = tarfile.open(filename, mode='r')
            tar.extractall()
            tar.close()
        else:
            print("IMDB archive directory already available")

        # collect and normalize test/train data
        print("cleaning up dataset ...")
        folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg', 'train/unsup']
        for fol in folders:
            newline = "\n".encode("utf-8")
            output = fol.replace('/', '-') + '.txt'
            # is there a better pattern to use
            txt_files = glob.glob(os.path.join(dirname, fol, '*.txt'))
            print(" %s: %i file" % (fol, len(txt_files)))
            with smart_open(os.path.join(dirname, output), "wb") as n:
                for i, txt in enumerate(txt_files):
                    with smart_open(txt, "rb") as t:
                        one_text = t.read().decode("utf-8")
                        for c in control_chars:
                            one_text = one_text.replace(c, ' ')
                        one_text = normalize_text(one_text)
                        all_lines.append(one_text)
                        n.write(one_text.encode("utf-8"))
                        n.write(newline)
        
        # save to disk for instant re-use any future run
        with smart_open(os.path.join(dirname, 'alldata-id.txt'), 'wb') as f:
            for idx, line in enumerate(all_lines):
                num_line = u"_*{0} {1}\n".format(idx, line)
                f.write(num_line.encode("utf-8"))
    
    assert os.path.isfile("./dataset/aclImdb/alldata-id.txt"), "alldata-id.txt unavailable"
    print("--SUCCESS--")
