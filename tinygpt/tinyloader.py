from tinygrad.tensor import Tensor

class TinyDataLoader:
    def __init__(self, dataset, batch_size=1) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            xs = []
            ys = []
            for i in range(self.batch_size):
                x, y = self.dataset[i]
                xs.append(x)
                ys.append(y)
            xs = Tensor.stack(*xs)
            ys = Tensor.stack(*ys)
            yield xs, ys
