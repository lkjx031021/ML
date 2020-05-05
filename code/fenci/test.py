import jieba


doc = '拉菲瑞特干红葡萄酒'
doc = '欧百乐私家特藏干红葡萄酒'

print(list(jieba.cut(doc)))
