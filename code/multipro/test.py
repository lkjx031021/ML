# coding:utf-8

import os
from multiprocessing import Process, Pool, Queue
import time
import random

def process_test(name):
    print('process_name:%s, pid:%s' % (name, os.getpid()))

def multi_test(name):
    print('process_name:{}, pid:{}'.format(name, os.getpid()))
    start = time.clock()
    time.sleep(random.random() * 3)
    end = time.clock()
    print('process_name:{}, run_time:{}'.format(name, end-start))


if __name__ == '__main__':
    print('process_id:%s' % os.getpid())

    p = Process(target=process_test, args=('name_1',))
    p.start()
    p.join()
    print('end_____')

    p = Pool(4)
    for i in range(5):
        p.apply_async(multi_test, args=(i,))

    print('Waiting for all subprocess done...')
    p.close()
    print('----------------------')
    print(os.getpid())
    # p.join()
    print('All subprocess done...')
    time.sleep(4)

    import subprocess

    print('$ nslookup www.python.org')
    r = subprocess.call(['nslookup', 'www.python.org'])
    print('Exit code:', r)

    import subprocess
    r = subprocess.call(['git', 'status'])
    print(r)