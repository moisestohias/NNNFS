from utils import MNIST, Batcher

mnist = Batcher(MNIST(Flat=False, OneHot=False))
for (x, y), _  in zip(mnist,range(1)): print(x.shape, y.shape)