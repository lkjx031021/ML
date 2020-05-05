#coding:utf-8

def merge_sort(ls):
    len_ = len(ls)
    if len_ <= 1:
        return ls
    mid = len_ // 2
    left = merge_sort(ls[:mid])
    right = merge_sort(ls[mid:])
    return merge(left, right)


def merge(left, right):
    c = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            c.append(left[i])
            i += 1
        else:
            c.append(right[j])
            j += 1

        if i == len(left):
            c.extend(right[j:])
        if j == len(right):
            c.extend(left[i:])

    return c

def quick_sort(ls):

    if len(ls) <= 1:
        return ls

    temp = ls[len(ls) // 2]
    left = [x for x in ls if x < temp]
    middle = [x for x in ls if x == temp]
    right = [x for x in ls if x > temp]
    return quick_sort(left) + middle + quick_sort(right)

print(quick_sort([2,6,9,1,4,2,8,]))

