# utils.py
import numpy as np

def batch_iterator(x: np.ndarray, y=None, MBS=32):
  n_sampels = x.shape[0]
  for i in range(0, n_sampels, MBS):
    begin, end = i, min(i+MBS, n_samples) # accounting for the case of the last iteration. (ZCR, AE..)
    if y: yield x[begin,end], y[begin,end]
    else: yield x[begin,end]

def shuffle_data(x, y, seed=0):
  if seed: np.random.seed(seed)
  idx = np.arange(x.shape[0]) # Only Shuffle the highest dim (shape[0])
  np.random.shuffle(idx)
  return x[idx], y[idx]

def shuffle_data_tuples(x, seed=None):
  """ Giving an ndarray of tuple of (input,target) data, Randomaly Shuffle it"""
  if seed: np.random.seed(seed)
  return np.random.shuffle(x)

def one_hot(a, classes=10): return np.eye(classes)[a] # .T Transpose if you don't want torch style
def one_hot_vector(a, classes=10): return np.eye(classes)[a].reshape(-1,classes,1) # extra empty dim to work with michs outptu.
def accuracy(X,Y): return t.sum(X-Y).item()/ len(X)

class Dataset:
  def __init__(self, xs, ys, Shuffle:bool=False, OneHot:bool=False, classes:int=None):
    if Shuffle: xs, ys = shuffle_data(xs, ys)
    self.xs = xs
    self.ys = one_hot(ys, classes) if OneHot and not isinstance(classes, int) else ys 
    self.counter = 0
    self.size = xs.shape[0]
  def __len__(self): return self.size
  def __iter__(self): return self
  def __next__(self):
    yld = self.xs[self.counter], self.ys[self.counter]
    if self.counter < self.size-1: self.counter += 1
    else: raise StopIteration
    return yld
  def __getitem__(self,n): return list(zip(self.xs[n], self.ys[n]))
  def __repr__(self): return f"{self.__class__.__name__}(xs, ys)"

  @classmethod
  def ImageFolder(cls, path):
    import os


class Batcher:
  """ Batcher is a DS wrapper to iterate over MBS. 
  Batcher should Shuffle since the DS load data once and batch for many epochs
  once the entire DS is exhuasted we create new batcher to reset counter + reshufle """
  def __init__(self, DS, MBS=128, Shuffle=True): 
    self.MBS = MBS
    self.DS = DS
    if Shuffle: DS.xs, DS.ys = shuffle_data(DS.xs, DS.ys)
    self.counter = 0
    self.size = len(DS)
  def __iter__(self): return self
  def __repr__(self): return f"{self.__class__.__name__}(xs, ys, {self.MBS})"
  def __next__(self):
    if self.counter >= self.size: raise StopIteration
    if self.size>self.counter+self.MBS: 
      batch = self.DS.xs[self.counter:self.counter+self.MBS], self.DS.ys[self.counter:self.counter+self.MBS]
    else: batch = self.DS.xs[self.counter:], self.DS.ys[self.counter:]
    self.counter += self.MBS
    return batch
  def __getitem__(self,n):
    if isinstance(n, slice): return list(zip(self.DS.xs[n], self.DS.ys[n]))
    elif isinstance(n, int): return self.DS.xs[n], self.DS.ys[n]
    else: raise TypeError(f"Index must be int or slice got {type(n)}")
  def __repr__(self): return f"{self.__class__.__name__}(xs, ys)"

class MNIST(Dataset):
  def __init__(self, Train=True, Validation=False, Flat=True, OneHot=True, **kw):
    (trainX, trainY), (valX,ValY), (TestX,TestY) = loadMnist(Flat=Flat, OneHot=OneHot)
    if Validation: super().__init__(valX, ValY, **kw)
    elif Train: super().__init__(trainX, trainY, **kw)
    else: super().__init__(TestX, TestY, **kw)


def splitDS(Size:int, Ratio:float = 0.8, Shuffle=True):
  rationIndx = int(Size*Ratio)
  Indx = np.random.permutation(Size) if Shuffle else np.arange(Size)
  return Indx[:rationIndx], Indx[rationIndx:]

# Mnist
def loadMnist(MNISTpath=None, Flat=True, Standardize=False, OneHot=False): # Mnist is already Flat and Normalized(0-1)
  """ This function should remain as it is, simple and self contained for quick testing (don't use Transform) """
  import requests, gzip, pickle, os
  if MNISTpath is None: MNISTpath = "/media/moises/D/DLDS" if os.name == "posix" else "D/DLDS" # Nix or Win
  url = 'https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz'
  if not os.path.exists(os.path.join(MNISTpath, "mnist.pkl.gz")):
    with open(os.path.join(MNISTpath, "mnist.pkl.gz"), "wb") as f:
      mnistPKLGZ = requests.get(url).content
      f.write(mnistPKLGZ)
  with gzip.open(os.path.join(MNISTpath, "mnist.pkl.gz"), "rb") as mn: 
    # ((50000, 784) (50000,)) ((10000, 784) (10000,)) ((10000, 784) (10000,))
    (xtr, ytr), (xva, yva), (xte, yte) = pickle.load(mn, encoding="latin-1") # tr, va, te
  if Standardize:
    xtr = xtr-xtr.mean(axis=1)[:,None] # Standardizing per-sample (!wrong:change this)
    xva = xva-xval.mean(axis=1)[:,None]
    xte = xte-xte.mean(axis=1)[:,None]
  if OneHot: ytr = one_hot(ytr, 10)
  if Flat: return (xtr, ytr), (xva, yva), (xte, yte) # tr, va, te 
  return (xtr.reshape(-1,1, 28, 28), ytr), (xva.reshape(-1,1, 28, 28), yva), (xte.reshape(-1,1, 28, 28), yte)

def insert_zeros(a, row, col):
    """ Insert zersos between elements which is used for the transposed conv"""
    for i in range(1, row+1): a = np.insert(a, np.arange(i, a.shape[-1], i), 0, axis=-1)
    for i in range(1, col+1): a = np.insert(a, np.arange(i, a.shape[-2], i), 0, axis=-2)
    return a

# Some Test:
if __name__ == '__main__':
  for x, y in Batcher(MNIST(OneHot=True), 32):
    print(x.shape, y.shape)
    break


