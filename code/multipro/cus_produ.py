# coding:utf-8

from multiprocessing import Pool, Queue, Process
import random, time, os

def write(q):
    print('Process to write:{}'.format(os.getpid()))
    for v in ['A', 'B', 'C']:
        print('Produce {}'.format(v))
        q.put(v)
        time.sleep(random.random() * 5)


def read(q):
    print('Process to read:{}'.format(os.getpid()))
    while 1:
        value = q.get(True)
        print('Get {} from queue'.format(value))


if __name__ == '__main__':
    q = Queue()
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    pw.start()
    pr.start()
    pw.join()
    pr.terminate()


