import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op='', requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self.shape = self.data.shape
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if not (self.requires_grad and other.requires_grad): return Tensor(self.data + other.data, (self, other), 'add')
        out = Tensor(self.data + other.data, (self, other), 'add', requires_grad=True)
        def _backward():
            """Do we have to check whether other requires grad or not? I don't think so, since other will take the gradient whether it need it or not, if doens't requires grad, then stop."""
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if not (self.requires_grad and other.requires_grad): return Tensor(self.data * other.data, (self, other), 'mul')
        out = Tensor(self.data * other.data, (self, other), 'mul')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo): v._backward()

    def dot(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if not (self.requires_grad and other.requires_grad): 
            print("What the hell")
            return Tensor(self.data.dot(other.data), (self, other), 'dot')
        out = Tensor(self.data.dot(other.data), (self, other), 'dot', requires_grad=True)
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    def __get__(self, indx): return self.data.__get__(indx)
    def __set__(self, indx, ): return self.data.__set__(indx)
    def __neg__(self): return self * -1 # -self
    def __radd__(self, other): return self + other # other + self
    def __sub__(self, other): return self + (-other) # self - other
    def __rsub__(self, other): return other + (-self) # other - self
    def __rmul__(self, other): return self * other # other * self
    def __truediv__(self, other): return self * other**-1 # self / other
    def __rtruediv__(self, other): return other * self**-1 # other / self
    def __repr__(self): return f"Tensor(data={self.data}, grad={self.grad})"

np.random.seed(5)
z = Tensor(np.random.randn(6,4))
w = Tensor(np.random.randn(4,2), requires_grad=True)
breakpoint()
a = z.dot(w)
grad = a.backward()
