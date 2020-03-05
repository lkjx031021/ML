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


if __name__ == '__main__':
    a = [14, 2, 34, 43, 21, 19]
    print (merge_sort(a))
