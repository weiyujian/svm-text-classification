svm文本分类,方法普通，主要是为了快速得到一个baseline

1.分词 语料格式：标签\t问句
python get_svm_input.py

2.训练
python svm_classification.py --input_file=xxx

3.测试
python svm_classification.py --is_train=false --input_file=xxx
