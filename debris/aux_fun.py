# import matplotlib.pyplot as plt
import numpy as np
from time import time

def load(input_file):
    data = open(input_file, 'r').read()     # should be simple plain text file
    chars = sorted(set(data))
    vocab_size = len(chars)
    ch2ix = {ch: i for i, ch in enumerate(chars)}
    ix2ch = {i: ch for i, ch in enumerate(chars)}
    return data, ch2ix, ix2ch, vocab_size

def encode(seq, vocab_size):                            # 1-of-k encoding
    enc = np.zeros((1, vocab_size), dtype=int)
    enc[0][seq[0]] = 1
    for i in range(1, len(seq)):
        row = np.zeros((1, vocab_size), dtype=int)
        row[0][seq[i]] = 1
        enc = np.append(enc, row, axis=0)
    return enc


def python_gen(data, seq_length, char_to_idx, vocab_size, p=0):
    p = int(p)
    print(p)
    while 1:
        if p + seq_length + 1 >= len(data):
            print("Aqui hemos llegado: ", p, len(data))
            p = 0  # go to start of data
        x = [char_to_idx[char] for char in data[p: p + seq_length]]  # Sequence of inputs (numbers)
        t = [char_to_idx[char] for char in data[1 + p: 1 + p + seq_length]]
        inputs = encode(x, vocab_size)  # shape: (seq_length, input_dim)
        targets = encode(t, vocab_size)
        p = p + seq_length
        yield inputs, targets

def tf_gen(data, seq_length, char_to_idx, vocab_size, p=0):
    p = int(p)
    while 1:
        if p + seq_length + 1 >= len(data):
            print("Aqui hemos llegado: ", p, len(data))
            p = 0  # go to start of data
        a = [char_to_idx[char] for char in data[p: p + seq_length]]  # Sequence of inputs (numbers)
        t = [char_to_idx[char] for char in data[1 + p: 1 + p + seq_length]]
        inputs = np.expand_dims(encode(a, vocab_size), axis=1)  # shape: (seq_length, input_dim)
        targets = np.expand_dims(encode(t, vocab_size), axis=1)
        p = p + seq_length
        yield inputs, targets


def keras_gen(data, seq_length, char_to_idx, vocab_size, p=0):
    p = int(p)
    while 1:
        if p + seq_length + 1 >= len(data):
            print("Aqui hemos llegado: ", p, len(data))
            p = 0  # go to start of data
        a = [char_to_idx[char] for char in data[p: p + seq_length]]  # Sequence of inputs (numbers)
        t = [char_to_idx[char] for char in data[1 + p: 1 + p + seq_length]]
        inputs = np.expand_dims(encode(a, vocab_size), axis=0)  # shape: (1, seq_length, input_dim)
        targets = np.expand_dims(encode(t, vocab_size), axis=0)
        # print(targets.shape)
        p = p + seq_length
        yield inputs, targets

