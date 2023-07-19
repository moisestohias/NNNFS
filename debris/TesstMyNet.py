# tesst.py

if __name__ == "__main__":
    # Z = np.random.randint(0,10, (2, 2, 6, 6))
    # Z = np.random.randn(MBS, 1, 28, 28)

    # C1 = Conv2d(Z.shape[1],10,5, inShape=Z.shape[-2:])
    # bottomGrad = np.random.randn(*z.shape)
    # C1.backward(bottomGrad)
    # print(C1.grads[0].shape, C1.grads[1].shape)


    model = Network()
    Act = Relu()
    C1 = Conv2d(1, 10, 5, inShape=(28,28))
    C2 = Conv2d(10, 6, 5, inShape=C1.outShape[-2:])
    C3 = Conv2d(6, 3, 5, inShape=C2.outShape[-2:])
    F  = Flatten(C3.outShape)
    D1 = Linear(F.outShape[-1],10)
    # D2 = Linear(D1.outShape[-1],10)
    SM = SoftmaxCELayer()

    model.add(C1)
    model.add(Act)
    model.add(C2)
    model.add(Act)
    model.add(C3)
    model.add(Act)
    model.add(F)
    model.add(D1)
    model.add(Act)
    # model.add(D2)
    # model.add(Act)
    model.add(SM)

    MBS = 32
    mnistTrain = Batcher(MNIST(Flat=False, OneHot=False), MBS)
    for x, y in mnistTrain:
        A = model.forward(x, y)
        model.backward()
        model.adam_trainstep()
        break

    #     correct = []
    #     for x, y in Batcher(MNIST(Validation=True), MBS):
    #         res = model.predict(x)
    #         correct.append(np.argmax(res, axis=1) == y)
    #     # print(len(correct))
    #     print(f'Validation accuracy: {np.mean(correct)}')
    #     print('-------------------------------------------------------')


