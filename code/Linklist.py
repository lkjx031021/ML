class LinkedListUnderflow(ValueError):
    pass

class Node:
    def __init__(self, elem, next_=None):
        self.elem = elem
        self.next_ = next_

class DNode(Node):
    def __init__(self, elem, prev=None, next_=None):
        Node.__init__(self, elem, next_)
        self.prev = prev

class LList:
    def __init__(self):
        self._head = None

    def is_empty(self):
        return self._head is None

    def prepend(self, elem):
        self._head = Node(elem, self._head)

    def pop(self):
        if self._head is None:
            raise LinkedListUnderflow("in pop")
        e = self._head.elem
        self._head = self._head.next_
        return e

    def pop_last(self):
        if self._head is None:
            raise LinkedListUnderflow("in pop_last")
        p = self._head
        if p is None:
            e = p.elem
            self._head = None
            return e
        while p.next_.next_ is not None:
            p = p.next_
        e = p.next_.elem
        p.next_ = None
        self.tail_ = p
        return e


    def append(self, elem):
        if self._head is None:
            self._head = Node(elem)
            return
        p = self._head
        while p.next_ is not None:
            p = p.next_
        p.next_ = Node(elem)
        self.tail_ = p.next_
        return

    def reversed(self):
        ''' 列表反转 '''
        if self.is_empty():return
        p = self._head
        while p.next_ is not None:
            temp = p.next_.next_
            p.next_.next_, self._head = self._head, p.next_
            p.next_ = temp


    def find(self, func):
        if self._head is None:
            raise LinkedListUnderflow("in find")
        p = self._head
        while p is not None:
            if func(p.elem):
                return p.elem
            p = p.next_

    def elements(self):
        p = self._head
        while p is not None:
            yield p.elem
            p = p.next_

    def fliter(self, func):
        p = self._head
        while p is not None:
            if func(p.elem):
                yield p.elem
            p = p.next_


class NewLList(LList):

    def __init__(self):
        LList.__init__()
        self.count = 0
        self.tail_ = None

    def prepend(self, elem):
        self._head = Node(elem, self._head)
        self.count += 1


llist1 = Node(1)
p = llist1

k = LList()

for i in range(10):
    k.append(i)
# for i in range(2):
#     k.append(i)

for i in k.elements():
    print(i)

print('==============')
k.reversed()
print('---------')
for i in k.elements():
    print(i)


a = [1,2,3]
