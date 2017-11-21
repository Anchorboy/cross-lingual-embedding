import cPickle
import numpy as np
from sklearn.datasets import make_classification

class AveragedPerceptron:
    def __init__(self, max_iter=15, n_classes=4, lr=1e-3):
        self.max_iter = max_iter
        self.n_classes = n_classes
        self.lr = lr

    def predict(self, x):
        x_pred = np.dot(self.W, np.append(x, 1))
        y_pred = np.argmax(x_pred)
        return y_pred

    def update_W(self, cnt, x, y_true, y_pred):
        """
        update W
        :param cnt: cnt
        :param x:
        :param y_true:
        :param y_pred:
        :return:
        """
        extend_x = np.append(x, 1)
        self.W[y_true] += self.lr * extend_x
        self.W[y_pred] -= self.lr * extend_x
        self.Wc[y_true] += self.lr * cnt * extend_x
        self.Wc[y_pred] -= self.lr * cnt * extend_x

    def fit(self, X, y):
        """
        fit data to the classifier
        :param X: input data, [n_samples x D-dimension], float32
        :param y: input labels, [n_samples], int32
        :return:
        """
        if isinstance(X, list) or isinstance(y, list):
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.int32)
        assert X.shape[0] == y.shape[0]
        n_samples, d = X.shape

        # init weight, bias
        self.W = np.zeros(shape=(self.n_classes, d + 1))
        self.Wc = np.zeros_like(self.W)

        cnt = 1
        for i in range(0, self.max_iter):
            correct_pred = 0
            rand_order = np.random.permutation(n_samples)
            for n in range(n_samples):
                x_, y_ = X[rand_order[n], :], y[rand_order[n]]
                y_pred = self.predict(x_)
                if y_pred == y_:
                    correct_pred += 1
                else:
                    self.update_W(cnt, x=x_, y_true=y_, y_pred=y_pred)
                cnt += 1
            acc = float(correct_pred) / n_samples
            print 'Iteration %d: acc = %f' % (i+1, acc*100)

        self.W -= 1.0 / cnt * self.Wc

    def evaluate(self, X, y):
        n_samples, d = X.shape
        acc = 0.0
        for n in range(n_samples):
            x_, y_ = X[n, :], y[n]
            y_pred = self.predict(x_)
            if y_pred == y_:
                acc += 1.0
        acc /= n_samples
        print 'Predict accuracy = %f' % (acc * 100)

    def save_weight(self):
        self.W.dump("W.npy")

    def load_weight(self):
        self.W = np.load("W.npy")

def train():
    clf = AveragedPerceptron(max_iter=50)
    X, y = make_classification(150, 20, n_informative=4, n_classes=4, n_clusters_per_class=4)
    clf.fit(X, y)
    clf.evaluate(X, y)
    clf.save_weight()

if __name__ == "__main__":
    train()