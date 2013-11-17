import numpy as np
import random


def randomWeights(numInputs, numOutputs):
    return np.random.random((numInputs + 1, numOutputs)) / 1000

def decode(number, base):
    output = np.zeros(base)
    output[number] = 1
    return output

def deltaRule(input, output, target, learningRate):
    '''Return the amounts to update the weights for one step of gradient descent.

    dw_ij = a * (t_j - y_j) * g_prime(h_j) * x_i
    where
        a = learning rate
        t_j = target output
        y_j = actual output
        g_prime = derivative of activation function
        h_j = weighted sum of inputs (g_prime(h_j) = 1 for perceptrons)
        x_i = ith input

    Args:
        input (n-length array): input to evaluate.
        output, target (m-length arrays): actual and desired outputs.

    Returns:
        an nxm array.
    '''
    return np.outer(input, target - output) * learningRate

def errors(outputs, targets):
    return (abs(output - target for output, target in zip(outputs, targets)))


class NeuralNet:
    '''Single-layer neural network.

    Attributes:
        weights (nxm array): n inputs, m outputs
    '''

    def __init__(self, weights):
        self.weights = weights

    def evaluate(self, input):
        '''
        Args: input (n-length array)
        Returns: an n-length array.
        '''
        return np.dot(self.weights.T, input)

    def updateWeights(self, input, target, learningRate):
        self.weights += deltaRule(input, self.evaluate(input), target, learningRate)

    def averageError(self, inputs, targets):
        return sum(sum(abs(self.evaluate(i) - t)) for i, t in zip(inputs, targets)) / len(inputs)

    # def errors(self, inputs, targets):
    #     # return (abs(self.evaluate(i) - t) for i, t in zip(inputs, targets))
    #     return errors((self.evaluate(i) for i in inputs), targets)

    def gradientDescent(self, inputs, targets, learningRate, convergenceThreshold):
        # If initial error is less than the convergence threshold, you've
        # already converged, so initialize prevError to 0.
        prevError = 0
        while True:
            error = self.averageError(inputs, targets)
            print(error)
            if abs(error - prevError) < convergenceThreshold:
                break
            prevError = error

            for input, target in zip(inputs, targets):
                self.updateWeights(input, target, learningRate)


def loadData(filepath):
    inputs = []
    targets = []
    with open(filepath) as f:
        for line in f:
            numbers = [int(n) for n in line.strip().split(',')]
            inputs.append(np.array([1] + numbers[:-1]))
            targets.append(np.array(decode(numbers[-1], base=10)))
    return inputs, targets


if __name__ == '__main__':

    path_train_100 = '/Users/frederick/Dropbox/data/handwriting/optdigits_100.tra'
    path_train = '/Users/frederick/Dropbox/data/handwriting/optdigits.tra'
    path_test = '/Users/frederick/Dropbox/data/handwriting/optdigits.tes'

    inputs_train_100, targets_train_100 = loadData(path_train_100)
    inputs_train, targets_train = loadData(path_train)
    inputs_test, targets_test = loadData(path_test)

    ann = NeuralNet(weights=randomWeights(64, 10))

    def train(inputs=inputs_train, targets=targets_train, learningRate=0.00001, convergenceThreshold=0.00001):
        ann.gradientDescent(inputs, targets, learningRate, convergenceThreshold)

    def test(inputs=inputs_test, targets=targets_test):
        score = numCorrect(inputs, targets)
        num = len(inputs)
        print('{} of {} inputs correctly evaluated ({:0.3f})%.'.format(
                score, num, score / num * 100))

    def check(input, target):
        '''Check if the output and target's max value's indices are the same.'''
        return ann.evaluate(input).argmax() == target.argmax()

    def numCorrect(inputs, targets):
        return sum(check(input, target) for input, target in zip(inputs, targets))
