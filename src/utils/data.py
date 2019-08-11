import os, re, json
import string
import glob
import tarfile
import numpy as np
import pandas as pd
from smart_open import smart_open


config = json.load(open('./config.json', 'r'))
data_path = config['data_path']
path_data_neg = os.path.join(data_path, config['dataset']['cam_neg'])
path_data_pos = os.path.join(data_path, config['dataset']['cam_pos'])
path_data_neg_tag = os.path.join(data_path, config['dataset']['cam_neg_tag'])
path_data_pos_tag = os.path.join(data_path, config['dataset']['cam_pos_tag'])
path_data_cam = os.path.join(data_path, config['dataset']['cam_data'])
path_IMDB = os.path.join(data_path, config['dataset']['IMDB'])
path_twitter = os.path.join(data_path, config['dataset']['twitter'])
path_douban = os.path.join(data_path, config['dataset']['douban'])

punctuations = string.punctuation+'``'+'"'


def load_data_from_file(sentiment):
    """
    Load data w/ POS tag given sentiment
    --
    # para sentiment: whether the review is neg or pos
    # return type: list(str)
    """
    path = path_data_neg if sentiment == 'neg' else path_data_pos
    files = os.listdir(path)
    reviews = []
    for file in files:
        if not os.path.isdir(file):
            f = open(os.path.join(path, file), 'r', encoding='utf-8')
            review = ""
            for line in f:
                review += line.lower().replace('\n', '')
            reviews.append(review)
            f.close()  # otherwise resource warning
    return reviews


def load_data_tag_from_file(sentiment):
    """
    Load data w/ POS tag given sentiment
    --
    # para sentiment: whether the review is neg or pos
    # return type: list(dict(str, str))
    """
    path = path_data_neg_tag if sentiment == 'neg' else path_data_pos_tag
    files = os.listdir(path)
    reviews_tags = []
    for file in files:
        if not os.path.isdir(file):
            f = open(os.path.join(path, file), 'r', encoding='utf-8')
            review_tag = dict()
            for line in f:
                word_tag = re.split(r'\t+', line[:-1])
                if len(word_tag) == 2 and word_tag[0] not in punctuations:
                    review_tag[word_tag[0].lower()] = word_tag[1]
            reviews_tags.append(review_tag)
            f.close()  # otherwise resource warning
    return reviews_tags


def prepare_data_camNLP():
    """
    prepare the Movie Review data (camNLP)
    <ONLY EXECUTED ONCE>
    """
    review_neg = load_data_from_file('neg')
    review_pos = load_data_from_file('pos')
    review_neg_label = np.stack((np.arange(len(review_neg)), np.array(review_neg), np.zeros(len(review_neg))), axis=-1)
    review_pos_label = np.stack((np.arange(len(review_neg)), np.array(review_pos), np.ones(len(review_pos))), axis=-1)    
    reviews = np.vstack((review_neg_label, review_pos_label))
    pd.DataFrame(reviews, columns=['index', 'review', 'sentiment']).to_csv(path_data_cam)


def prepare_data_IMDB():
    """
    prepare the IMDB data (normalization and cleaning)
    <ONLY EXECUTED ONCE>
    """
    dirname = os.path.join(data_path, 'aclImdb')
    filename = os.path.join(data_path, 'aclImdb_v1.tar.gz')
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

    if not os.path.isfile(path_IMDB):
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
                for _, txt in enumerate(txt_files):
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
    
    assert os.path.isfile(path_IMDB), "alldata-id.txt unavailable"
    print("--SUCCESS--")
