import os
import json
import numpy as np
import pandas as pd
from src.utils.display import print_new


class loadPretrain(object):
    def __init__(self, chunksize=100):
        self.config = json.load(open('./config.json', 'r', encoding='utf-8'))
        self.embed_en_filepath = ""
        self.embed_zh_filepath = os.path.join(self.config['data_path'],
                                            self.config['pretrain']['filepath'])
        self.chunksize = chunksize
    
    def load_embeddings(self):
        count = 0
        self.data_zh = pd.DataFrame()
        with open(self.embed_zh_filepath, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                embeds = line.split(' ')
                print(len(embeds), embeds[0], end='\t')
                # self.data_zh = pd.concat([self.data_zh, chunk], ignore_index=True)
                # print(self.data_zh.head())
                # print(self.data_zh.loc[1])
                count += 1
                if count == 500:
                    break