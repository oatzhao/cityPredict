#coding = utf-8
import time
from concurrent.futures import ThreadPoolExecutor

#死锁
def wait_on_b():
    time.sleep(5)
    #print(b.result())  # b不会完成，他一直在等待a的return结果
    return 5


def wait_on_a():
    time.sleep(5)
    #print(a.result())  # 同理a也不会完成，他也是在等待b的结果
    return 6


executor = ThreadPoolExecutor(max_workers=2)
a = executor.submit(wait_on_b)
b = executor.submit(wait_on_a)