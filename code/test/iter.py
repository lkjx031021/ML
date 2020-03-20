
class Fibs(object):

    def __init__(self):
        self.a = 0
        self.b = 1

    def __next__(self):
        self.a, self.b = self.b, self.a+self.b
        return self.a

    def __iter__(self):
        print(1111)
        return  self


fib = Fibs()
print(next(fib))
print(next(fib))
print(next(fib))
print(next(fib))

exit()
exit()
for i in fib:
    if i > 1000:
        print(i)
        break

