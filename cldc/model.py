import os
import cPickle
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class AveragedPerceptron:
    def __init__(self, max_iter=15, n_classes=4, lr=5e-3):
        self.max_iter = max_iter
        self.n_classes = n_classes
        self.lr = lr
        self.paren_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        self.util_path = os.path.join(self.paren_path, 'data', 'utils')

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

    def fit(self, X, y, X_valid=None, y_valid=None, verbose=False):
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
        if not hasattr(self, 'W'):
            self.W = np.zeros(shape=(self.n_classes, d + 1))
        else:
            self.W += 1.0 / self.cnt * self.Wc

        if not hasattr(self, 'Wc'):
            self.Wc = np.zeros_like(self.W)

        if not hasattr(self, 'cnt'):
            self.cnt = 1
        best_train_scores = []
        best_valid_scores = []
        for i in range(0, self.max_iter):
            correct_pred = 0
            rand_order = np.random.permutation(n_samples)
            for n in range(n_samples):
                x_, y_ = X[rand_order[n], :], y[rand_order[n]]
                y_pred = self.predict(x_)
                if y_pred == y_:
                    correct_pred += 1
                else:
                    self.update_W(self.cnt, x=x_, y_true=y_, y_pred=y_pred)
                self.cnt += 1
            acc = float(correct_pred) / n_samples
            best_train_scores += [acc]
            if verbose:
                print 'Iteration %d: acc = %f' % (i+1, acc*100)

            if not X_valid is None and not y_valid is None:
                best_valid_scores += [self.evaluate(X_valid, y_valid)]

        self.W -= 1.0 / self.cnt * self.Wc
        return best_train_scores, best_valid_scores


    def evaluate(self, X, y, verbose=False):
        if isinstance(X, list):
            X = np.asarray(X, dtype=np.float32)
        n_samples, d = X.shape
        acc = 0.0
        for n in range(n_samples):
            x_, y_ = X[n, :], y[n]
            y_pred = self.predict(x_)
            if y_pred == y_:
                acc += 1.0
        acc /= n_samples
        if verbose:
            print 'Predict accuracy = %f' % (acc * 100)
        return acc

    def save_weight(self, lang, embed_name):
        out_path = os.path.join(self.util_path, lang, embed_name + "_weight.npy")
        self.W.dump(out_path)

    def load_weight(self, lang, embed_name):
        in_path = os.path.join(self.util_path, lang, embed_name + "_weight.npy")
        self.W = np.load(in_path)

def train():
    clf = AveragedPerceptron(max_iter=10)
    X, y = make_classification(1500, 50, n_informative=4, n_classes=4, n_clusters_per_class=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf.fit(X_train, y_train, X_test, y_test)
    # clf.evaluate(X_train, y_train)
    # clf.evaluate(X_test, y_test)
    clf.save_weight("test","test")

if __name__ == "__main__":
    train()