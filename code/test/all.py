# coding:utf-8

def all_p(result, str, ls):
    if len(ls) == 1:
        result.append(str+ls[0])

    for ftr in ls:
        temp_ls = ls[:]
        temp_ls.remove(ftr)
        all_p(result, str+ftr,temp_ls)



test = []
ls = ['a', 'b', 'c']
all_p(test, '', ls)
print(test)