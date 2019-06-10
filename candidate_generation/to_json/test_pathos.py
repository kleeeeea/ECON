import pathos
import numpy as np
import time
import spotlight

def squared(x):
    np
    time.sleep(.5)
    return x ** 2


if __name__ == '__main__':
    x = np.arange(400).reshape(50, 8)
    p = pathos.pools.ProcessPool()
    t = pathos.pools.ThreadPool()

    st = time.time()
    ans = [squared(i) for i in x]
    et = time.time()
    print(et-st)

    st = time.time()
    ans = p.map(squared, x)
    et = time.time()
    print(et-st)

    st = time.time()
    ans = p.imap(squared, x)
    list(ans)
    et = time.time()
    print(et-st)

    st = time.time()
    ans = p.uimap(squared, x)
    list(ans)
    et = time.time()
    print(et-st)

    st = time.time()
    ans = t.map(squared, x)
    et = time.time()
    print(et-st)

    st = time.time()
    ans = t.imap(squared, x)
    list(ans)
    et = time.time()
    print(et-st)

    st = time.time()
    ans = t.uimap(squared, x)
    list(ans)
    et = time.time()
    print(et-st)