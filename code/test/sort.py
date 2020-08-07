#coding:utf-8

# def merge(iform, ito, low, mid, high):
#     i, j, k = low, mid, low
#     while i < mid and j < high:
#         # if iform[i].key <= iform[j].key:
#         if iform[i] <= iform[j]:
#             ito[k] = iform[i]
#             i += 1
#         else:
#             ito[k] = iform[i]
#             j += 1
#         k += 1
#
#     while i < mid:
#         ito[k] = iform[i]
#         i += 1
#         k += 1
#     while j < high:
#         ito[k] = iform[j]
#         j += 1
#         k += 1
#
# def merge_pass(iform, ito, ilen, slen):
#     i = 0
#     while i + 2*slen < ilen:
#         merge(iform, ito, i, i+slen, i + 2*slen)
#         i += 2 * slen
#     if i + slen < ilen:
#         merge(iform, ito, i, i + slen, ilen)
#     else:
#         for j in range(i, ilen):
#             ito[j] = iform[j]
#
#
# def merge_sort(lst):
#     slen, ilen = 1, len(lst)
#     templst = [None] * ilen
#     while slen < ilen:
#         merge_pass(lst, templst, ilen, slen)
#         slen *= 2
#         merge_pass(templst, lst, ilen, slen)
#         slen *= 2


a = [64,3,2,1,6,19,12,33]


def merge(a, b):
    print(a, b)
    c = []
    h = j = 0
    while j < len(a) and h < len(b):
        if a[j] < b[h]:
            c.append(a[j])
            j += 1
        else:
            c.append(b[h])
            h += 1

    if j == len(a):
        for i in b[h:]:
            c.append(i)
    else:
        for i in a[j:]:
            c.append(i)

    return c


def merge_sort(lists):
    if len(lists) <= 1:
        return lists
    middle = len(lists)//2
    left = merge_sort(lists[:middle])
    right = merge_sort(lists[middle:])
    return merge(left, right)

def permutation(result, str, list):
    """
        取一个数组的全排列
        list：为输入列表
        str：传空字符串
        result： 为结果列表
    """
    print(result, str, list[0])
    if len(list) == 1:
        result.append(str + "," + list[0])
    else:
        for temp_str in list:
            temp_list = list[:]
            temp_list.remove(temp_str)
            # print temp_list
            permutation(result, str + "," + temp_str, temp_list)


def quick_sort(ls):
    if len(ls) <=1:
        return ls

    mid = len(ls) // 2
    mid_elem = ls[mid]
    left_ls = []
    right_ls = []
    mid_ls = []
    for elem in ls:
        if elem < mid_elem:
            left_ls.append(elem)
        elif elem == mid_elem:
            mid_ls.append(elem)
        else:
            right_ls.append(elem)
    print('----------')
    print(left_ls, mid_ls, right_ls)

    return quick_sort(left_ls) + mid_ls + quick_sort(right_ls)



if __name__ == '__main__':
    a = [14, 2, 34, 18, 43, 21, 19]
    print (merge_sort(a))
    print(quick_sort(a))
