import jieba

def get_seg_list(query):
    seg_list = jieba.cut(query)
    return seg_list

def prepare_data(file_name):
    fw = open(file_name + ".svm_seg", "w")
    with open(file_name) as f:
        for line in f:
            tmp_list = line.strip().split('\t')
            if len(tmp_list) != 2:continue
            query = tmp_list[1]
            label = tmp_list[0]
            query_seg_list = get_seg_list(query)
            query_seg = ' '.join(query_seg_list).encode('utf-8')
            fw.write(query+"\t"+query_seg+"\t"+label+"\n")
    fw.close()
    return 

if __name__=='__main__':
    prepare_data("./data/cnews.train.debug")
