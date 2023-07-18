import numpy as np

def one_hot(a, classes=10): return np.eye(classes)[a].reshape(-1,10,1)

def loadMnist(path=None):
    import pickle, gzip, os
    if path is None: 
        path = "/media/moises/D/DLDS" if os.name == "posix" else "D/DLDS" # Nix or Win
    url = 'https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz'
    if not os.path.exists(os.path.join(path, "mnist.pkl.gz")):
        with open(os.path.join(path, "mnist.pkl.gz"), "wb") as f:
            mnistPKLGZ = requests.get(url).content
            f.write(mnistPKLGZ)
    with gzip.open(os.path.join(path, "mnist.pkl.gz"), "rb") as mn: 
        tr, va, te = pickle.load(mn, encoding="latin-1")
    return tr, va, te


def handleMnist(tr, va, te):
    tr, va, te = loadMnist()
    tr = zip(tr[0].reshape(-1, 784, 1), one_hot(tr[1]))
    va = zip(va[0].reshape(-1, 784, 1), va[1])
    te = zip(te[0].reshape(-1, 784, 1), te[1])
    return tr, va, te



# Act
def sig(z): return 1.0/(1.0+np.exp(-z))
def sigP(z): s = sig(z) ; return s*(1-s)

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def forward(self, a):
        for b, w in zip(self.biases, self.weights): a = sig(np.dot(w, a)+b)
        return a

    def SGD(self, trD, epochs, MBS, LR,
            test_data=None):
        trD = list(trD)
        n = len(trD)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            np.random.shuffle(trD)
            mini_batches = [trD[k:k+MBS] for k in range(0, n, MBS)]
            for MB in mini_batches: self.update_mini_batch(MB, LR)
            if test_data: print(f"Epoch {j} : {self.evaluate(test_data)} / {n_test}")
            else: print(f"Epoch {j} complete")

    def update_mini_batch(self, MB, LR):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in MB:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(LR/len(MB))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(LR/len(MB))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        act = x
        acts = [x] 
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, act)+b
            zs.append(z)
            act = sig(z)
            acts.append(act)
        # backward pass
        delta = self.costP(acts[-1], y) * sigP(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, acts[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigP(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, acts[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.forward(x)), y) for (x, y) in test_data]
        return sum((x == y) for (x, y) in test_results)

    def costP(self, output_acts, y): return (output_acts-y)

tr, va, te = loadMnist()
tr, va, te = handleMnist(tr, va, te)
net = Network([784, 30, 20, 10])
epochs, MBS, LR = 4, 20, 2
net.SGD(tr, epochs, MBS, LR, te)