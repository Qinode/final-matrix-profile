import numpy as np


def get_chain(li, ri, start):
    chain = []
    while ri[start] != -1 and li[ri[start]] == start:
        chain.append(start)
        start = ri[start]

    return chain


def get_brave_chain(ri, start):
    chain = []
    while start != -1:
        chain.append(start)
        start = ri[start]

    return chain


def get_brave_chain_backward(li, start):
    chain = []
    while start != -1:
        chain.append(start)
        start = li[start]

    return chain


def get_longest_chain(li, ri):
    length = li.shape[0]

    l = np.ones((length, 1))

    for i in range(length):
        l[i] = len(get_chain(li, ri, i))

    return np.argmax(l)

