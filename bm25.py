#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: bm25.py
@time:2022/04/16
@description:
"""
import math
import os
import jieba
import pickle
import logging

jieba.setLogLevel(log_level=logging.INFO)


class BM25Param(object):
    def __init__(self, f, df, idf, length, avg_length, docs_list, line_length_list,k1=1.5, b=0.75):
        self.f = f
        self.df = df
        self.k1 = k1
        self.b = b
        self.idf = idf
        self.length = length
        self.avg_length = avg_length
        self.docs_list = docs_list
        self.line_length_list = line_length_list

    def __str__(self):
        return f"k1:{self.k1}, b:{self.b}"


class BM25(object):
    _param_pkl = "data/param.pkl"
    _docs_path = "data/data.txt"
    _stop_words_path = "data/stop_words.txt"
    _stop_words = []

    def __init__(self, docs=""):
        self.docs = docs
        self.param: BM25Param = self._load_param()

    def _load_stop_words(self):
        if not os.path.exists(self._stop_words_path):
            raise Exception(f"system stop words: {self._stop_words_path} not found")
        stop_words = []
        with open(self._stop_words_path, 'r', encoding='utf8') as reader:
            for line in reader:
                line = line.strip()
                stop_words.append(line)
        return stop_words

    def _build_param(self):

        def _cal_param(reader_obj):
            f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
            df = {}  # 存储每个词及出现了该词的文档数量
            idf = {}  # 存储每个词的idf值
            lines = reader_obj.readlines()
            length = len(lines)
            words_count = 0
            docs_list = []
            line_length_list =[]
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                words = [word for word in jieba.lcut(line) if word and word not in self._stop_words]
                line_length_list.append(len(words))
                docs_list.append(line)
                words_count += len(words)
                tmp_dict = {}
                for word in words:
                    tmp_dict[word] = tmp_dict.get(word, 0) + 1
                f.append(tmp_dict)
                for word in tmp_dict.keys():
                    df[word] = df.get(word, 0) + 1
            for word, num in df.items():
                idf[word] = math.log(length - num + 0.5) - math.log(num + 0.5)
            param = BM25Param(f, df, idf, length, words_count / length, docs_list, line_length_list)
            return param

        # cal
        if self.docs:
            if not os.path.exists(self.docs):
                raise Exception(f"input docs {self.docs} not found")
            with open(self.docs, 'r', encoding='utf8') as reader:
                param = _cal_param(reader)

        else:
            if not os.path.exists(self._docs_path):
                raise Exception(f"system docs {self._docs_path} not found")
            with open(self._docs_path, 'r', encoding='utf8') as reader:
                param = _cal_param(reader)

        with open(self._param_pkl, 'wb') as writer:
            pickle.dump(param, writer)
        return param

    def _load_param(self):
        self._stop_words = self._load_stop_words()
        if self.docs:
            param = self._build_param()
        else:
            if not os.path.exists(self._param_pkl):
                param = self._build_param()
            else:
                with open(self._param_pkl, 'rb') as reader:
                    param = pickle.load(reader)
        return param

    def _cal_similarity(self, words, index):
        score = 0
        for word in words:
            if word not in self.param.f[index]:
                continue
            molecular = self.param.idf[word] * self.param.f[index][word] * (self.param.k1 + 1)
            denominator = self.param.f[index][word] + self.param.k1 * (1 - self.param.b +
                                                                       self.param.b * self.param.line_length_list[index] /
                                                                       self.param.avg_length)
            score += molecular / denominator
        return score

    def cal_similarity(self, query: str):
        words = [word for word in jieba.lcut(query) if word and word not in self._stop_words]
        score_list = []
        for index in range(self.param.length):
            score = self._cal_similarity(words, index)
            score_list.append((self.param.docs_list[index], score))
        return score_list


if __name__ == '__main__':
    bm25 = BM25()
    result = bm25.cal_similarity("自然语言处理并不是一般地研究自然语言")
    for line, score in result:
        print(line, score)
