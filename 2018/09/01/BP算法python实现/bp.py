import numpy as np

def loss(pred, y):
    return np.sum((pred - y) ** 2)

def loss_prime(pred, y):
    return pred - y

class network:

    def __init__(self, input_size, hidden_size, num_layers, output_size, loss = loss, loss_prime = loss_prime):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # activation function
        self.activation = self.sigmoid
        # derivative of activation function
        self.activation_prime = self.sigmoid_prime
        # loss funciton
        self.loss = loss
        # derivative of loss function
        self.loss_prime = loss_prime

        # input->hidden
        self.w_ih = np.random.randn(input_size, hidden_size)
        self.b_ih = np.random.randn(1, hidden_size)

        # hidden layers
        self.W_hh = [np.random.randn(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        self.B_hh = [np.random.randn(1, hidden_size) for _ in range(num_layers - 1)]

        # hidden->output
        self.w_ho = np.random.randn(hidden_size, output_size)
        self.b_ho = np.random.randn(1, output_size)

        # assemble w and b
        self.W = [self.w_ih]
        self.W.extend(self.W_hh)
        self.W.append(self.w_ho)

        self.B = [self.b_ih]
        self.B.extend(self.B_hh)
        self.B.append(self.b_ho)

    # activation
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # forward pass, calculate the output of the network
    def forward(self, a):
        for w, b in zip(self.W, self.B):
            a = self.activation(np.dot(a, w) + b)
        return a

    # backpropagate error
    def backward(self, x, y):
        delta_w = [np.zeros(w.shape) for w in self.W]
        delta_b = [np.zeros(b.shape) for b in self.B]

        # get output of each layer in forward pass
        out = x
        outs = []
        zs = []
        for w, b in zip(self.W, self.B):
            z = np.dot(out, w) + b
            zs.append(z)
            out = self.activation(z)
            outs.append(out)

        # δ of last layer
        delta = self.loss_prime(outs[-1], y) * self.activation_prime(zs[-1])

        delta_b[-1] = delta
        delta_w[-1] = np.dot(outs[-2].transpose(), delta)

        for i in range(2, len(delta_w)):
            delta = np.dot(delta, self.W[-i+1].transpose()) * self.activation_prime(zs[-i])
            delta_b[-i] = delta
            delta_w[-i] = np.dot(outs[-i-1].transpose(), delta)

        return delta_w, delta_b

    # update w and b
    def update(self, batch, lr):
        delta_w = [np.zeros(w.shape) for w in self.W]
        delta_b = [np.zeros(b.shape) for b in self.B]

        for x, y in batch:
            d_w, d_b = self.backward(x, y)
            delta_w = [a + b for a, b in zip(delta_w, d_w)]
            delta_b = [a + b for a, b in zip(delta_b, d_b)]

        self.W = [w - lr * t for w, t in zip(self.W, delta_w)]
        self.B = [b - lr * t for b, t in zip(self.B, delta_b)]

    # SGD training
    def train(self, train_data, epochs, batch_size, lr):
        for i in range(epochs):
            np.random.shuffle(train_data)
            batches = [train_data[t : t + batch_size] for t in range(0, len(train_data), batch_size)]

            for batch in batches:
                self.update(batch, lr)

            loss = 0
            for x, y in train_data:
                loss += self.loss(self.forward(x), y)
            loss /= len(train_data)

            print("Epoch %d done, loss: %f" % (i + 1, loss))

    # predict
    def predict(self, x):
        return self.forward(x)


# use it for handwriting digits classification
import tensorflow as tf
mnist = tf.keras.datasets.mnist

def onehot(y):
    arr = np.zeros([y.shape[0], 10])
    for i in range(y.shape[0]):
        arr[i][y[i]] = 1
    return arr

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape([-1, 28 * 28])
x_test = x_test.reshape([-1, 28 * 28])
y_train = onehot(y_train)
y_test = onehot(y_test)

train_data = [t for t in zip(x_train, y_train)]
test_data = [t for t in zip(x_test, y_test)]

input_size = 28 * 28
hidden_size = 100
num_layers = 3
output_size = 10

net = network(input_size, hidden_size, num_layers, output_size)

lr = 0.005
epochs = 100
batch_size = 100

net.train(train_data, epochs, batch_size, lr)

def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)

correct = 0
for x, y in test_data:
    ret = net.forward(x)
    pred = softmax(ret)
    if np.argmax(pred) == np.argmax(y):
        correct += 1

acc = float(correct) / len(test_data)
print('test accuracy: ', acc)
