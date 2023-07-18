class Tensor(np.ndarray):
  def __init__(self, data, _parrents=(), _op='', requires_grad=False):
    self.data = data
    self.requires_grad = requires_grad
    self.isLeaf = True
    self.grad = 0
    # internal variables used for autograd graph construction
    self._backward = lambda: None
    self._prev = set(_parrents)
    self._gradFun = None
    self._op = _op # the op that produced this node, for graphviz / debugging / etc

  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, (self, other), '+')
    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward
    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Tensor(other)
    out = Tensor(self.data * other.data, (self, other), '*')
    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Tensor(self.data**other, (self,), f'**{other}')
    def _backward(): self.grad += (other * self.data**(other-1)) * out.grad
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
      for v in reversed(topo):
          v._backward()

  def __neg__(self): return self * -1 # -self 
  def __radd__(self, other): return self + other # other + self 
  def __sub__(self, other): return self + (-other) # self - other 
  def __rsub__(self, other): return other + (-self) # other - self 
  def __rmul__(self, other): return self * other # other * self 
  def __truediv__(self, other): return self * other**-1 # self / other 
  def __rtruediv__(self, other): return other * self**-1 # other / self 
  def __repr__(self): return f"Tensor(data={self.data}, grad={self.grad})"