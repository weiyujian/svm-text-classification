#!/usr/bin/python
#-*-coding:utf-8-*-

import pdb
import numpy as np
import cPickle
from nltk.util import ngrams as ingrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import sys
import argparse

def get_ngram(text, max_len, stop_set):
    word_list = []
    for inx in xrange(1,max_len+1,1):
        gram_list = generate_ngram(text,stop_set,inx)
        word_list += [word for word in gram_list if word not in stop_set]
    word_list += generate_skipgram(text, stop_set)
    return " ".join(word_list)

def generate_ngram(text,stop_set,ngram_len):
    words = text.split(" ")
    ngram_list = []
    for wlist in ingrams(words, ngram_len, pad_right=True):
        if wlist[0] is None:continue
        skip = False
        w_inx = 0
        while w_inx < ngram_len:
            if wlist[w_inx] is None or wlist[w_inx] in stop_set:
                skip=True
                break
            w_inx +=1
        if skip:continue
        ngram_list.append("_".join(wlist))
    return ngram_list

def generate_skipgram(text,stop_set,ngram_len=16):
    words = text.split(" ")
    pair_set = set()
    for inx in xrange(len(words)):
        if words[inx] in stop_set:continue
        for iny in xrange(1,ngram_len,1):
            if inx + iny >= len(words):break
            if words[inx+iny] in stop_set:continue
            pair_set.add(words[inx]+"|"+words[inx+iny])
    return list(pair_set)

def read_label(file_name):
    f = open(file_name)
    label_map = {}
    label_map_inv = {}
    data_list = []
    check = set([])
    idx = 0
    for line in f:
        tmp_list = line.strip().split("\t")
        if len(tmp_list) != 3:continue
        query_raw = tmp_list[0]
        query_seg = tmp_list[1]
        label_raw = tmp_list[-1]
        if label_raw not in label_map:
            label_map[label_raw] = idx
            label_map_inv[idx] = label_raw
            idx += 1
        if query_seg not in check:
            check.add(query_seg)
            data_list.append((query_seg, label_map[label_raw]))
    f.close()
    return label_map, label_map_inv, data_list

def read_test(file_name, label_map):
    f = open(file_name)
    data_list = []
    for line in f:
        tmp_list = line.strip().split("\t")
        if len(tmp_list) != 3:continue
        query_raw = tmp_list[0]
        query_seg = tmp_list[1]
        label_raw = tmp_list[-1]
        if label_raw not in label_map:continue
        data_list.append((query_seg, label_map[label_raw]))
    f.close()
    return data_list 

def prepare_train_data(data_list, stop_set=set([])):
    """make train data"""
    x_train = []
    y_train = []
    for inx in xrange(len(data_list)):
        query_seg, label = data_list[inx]
        text = get_ngram(query_seg, 2, stop_set)#get nram and skip gram str
        #中间还能做点特征泛化的事情，比如数字泛化，英文泛化这些
        x_train.append(text)
        y_train.append(label)
    return [x_train, y_train]

def get_tfidf_feature(texts):
    #accordding to train data, fit tfidf model
    #vectorizer = TfidfVectorizer(sublinear_tf=False, token_pattern=r"(?u)\b\w+\b", min_df=1)#token_pattern=r'\S+'意思是将空格视为切分符
    vectorizer = TfidfVectorizer(sublinear_tf=False, token_pattern=r'\S+', min_df=1)
    vectorizer.fit(texts)
    return vectorizer

def randomly_shuffile_data(data_list):
    #data_list: [(query_seg, label),()...]
    np.random.seed(10)
    dev_sample_percentage = 0.1
    shuffle_indices = np.random.permutation(np.arange(len(data_list)))
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(data_list)))
    
    train_data_list = []
    dev_data_list = []
    for inx in shuffle_indices[:dev_sample_index]:
        train_data_list.append(data_list[inx])

    for inx in shuffle_indices[dev_sample_index:]:
        dev_data_list.append(data_list[inx])
    return train_data_list, dev_data_list
    
def feature_transform_predict(x, vectorizer, clf):
    x = vectorizer.transform(x)
    y_pred = clf.predict(x)
    y_score = clf.decision_function(x)
    return y_pred, y_score

def train(train_file,model_version,stop_set):
    train_label_map, train_label_map_inv, train_list_raw = read_label(train_file)
    train_list, test_list = randomly_shuffile_data(train_list_raw)
    x_train, y_train = prepare_train_data(train_list, stop_set)
    x_dev, y_dev = prepare_train_data(test_list, stop_set)
    vectorizer = get_tfidf_feature(x_train)
    x_train = vectorizer.transform(x_train)
    clf = LinearSVC(C=0.8)
    clf.fit(x_train, y_train)
    cPickle.dump([clf, vectorizer, train_label_map, train_label_map_inv], open(model_version, "wb"))
    
    y_dev_pred, y_dev_score = feature_transform_predict(x_dev, vectorizer, clf)
    y_dev = np.array(y_dev)
    print "dev accuracy:",accuracy_score(y_dev, y_dev_pred)
    return

def predict_batch(test_file, model_version, stop_set):
    fw = open(test_file+".out","w")
    clf, vectorizer, train_label_map, train_label_map_inv = cPickle.load(open(model_version,"rb"))
    test_list = read_test(test_file, train_label_map)
    x_test, y_test = prepare_train_data(test_list, stop_set)
    y_test_pred, score = feature_transform_predict(x_test, vectorizer, clf)
    y_test = np.array(y_test)
    print(classification_report(y_test, y_test_pred))
    total_cnt = len(test_list)
    acc_cnt = 0
    print "test accuracy:", accuracy_score(y_test, y_test_pred)
    for i in range(total_cnt):
        fw.write(test_list[i][0]+"\t"+str(y_test[i])+"\t"+str(y_test_pred[i])+"\t"+str(score[i][y_test_pred[i]])+"\n")
        if y_test_pred[i] == y_test[i]:
            acc_cnt += 1
    acc = 1.0*acc_cnt/total_cnt
    print "acc_cnt:",acc_cnt
    print "total_cnt:",total_cnt
    print "acc:",acc
    fw.write("acc:"+"\t"+str(acc)+"\t"+"total_cnt:"+"\t"+str(total_cnt)+"\n")
    fw.close()
    return
       
if __name__=='__main__':
    stop_set = set([])
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', default='svm_model_v1',type=str, help='save trained model')
    parser.add_argument('--is_train', default='true',type=str, help='train or test mode')
    parser.add_argument('--input_file', default='./data/cnews.train.txt.svm_seg',type=str, help='train or test file')
    args = parser.parse_args()
    model_version = args.model_version

    is_train = args.is_train
    is_train = True if is_train.lower().startswith('true') else False
    input_file = args.input_file
    if is_train:
        train(input_file, model_version, stop_set)
    else:
        predict_batch(input_file, model_version, stop_set)
