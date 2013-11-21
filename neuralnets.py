import random

import numpy as np


def randomWeights(numInputs, numOutputs):
    '''Return a (numInputs + 1) x numOutputs array of small random numbers.

    numInputs should not include the bias input.
    '''
    return np.random.random((numInputs + 1, numOutputs)) / 1000


def decode(number, base):
    '''Return a <base>-length array of zeros, with a 1 at index <number>.'''
    output = np.zeros(base)
    output[number] = 1
    return output


def deltaRule(input, output, target, learningRate):
    '''Return the weight updates for one step of gradient descent.

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
        self.numUpdates = 0

    def evaluate(self, input):
        '''
        Args: input (n-length array)
        Returns: an m-length array.
        '''
        return np.dot(self.weights.T, input)

    def updateWeights(self, input, target, learningRate):
        self.numUpdates += 1
        self.weights += deltaRule(input, self.evaluate(input), target,
                                  learningRate)

    def totalError(self, input, target):
        '''Calculate the total error over the output nodes for the given input.

        Args:
            input (n-length array)
            target (m-length array)

        Returns:
            The sum of the absolute differences between each output node and
            the corresponding target output value.
        '''
        # Note that the builtin abs works just fine on numpy arrays.
        return sum(abs(self.evaluate(input) - target))

    def averageError(self, data):
        '''Calculate the average error in the NeuralNet's evaluation of the
        given data.

        Args:
            data (i-length sequence of (n-length array, m-length array)):
                The input data to be evaluated and target outputs.

        Returns:
            The average difference of the NeuralNet's outputs from the target
            outputs.
        '''
        return sum(self.totalError(*datum) for datum in data) / len(data)

    # def errorChange(self, data):
    #     prevError = 0
    #     while True:
    #         error = self.averageError(data)
    #         yield error
    #         prevError = error

    def gradientDescent(self, data, learningRate, convergenceThreshold):
        '''Train the NeuralNet using gradient descent on the given data.

        Continue training on all the data until the change in error between
        training rounds is less than convergenceThreshold.

        Args:
            data (i-length sequence of (n-length input array, m-length target
                array))
            learningRate (0 < value < 1)
            convergenceThreshold
        '''
        # Loop until the change in error is less than convergenceThreshold.
        prevError = 0
        while True:
            error = self.averageError(data)
            if abs(error - prevError) < convergenceThreshold:
                break
            prevError = error

            for input, target in data:
                self.updateWeights(input, target, learningRate)


class Classifier(NeuralNet):

    def check(self, input, target):
        '''Check if the output and target evaluate to the same label.'''
        return self.evaluate(input).argmax() == target.argmax()

    def numCorrect(self, data):
        return sum(self.check(input, target) for input, target in data)


def readData(filepath):
    '''
    Yields:
        (data, label)
    '''
    with open(filepath) as f:
        for line in f:
            numbers = [int(n) for n in line.strip().split(',')]
            yield numbers[:-1], numbers[-1]


def translate(input, label):
    '''
    Args:
        input ((n-1)-length sequence)
        label: classification of the input.
    Returns:
        n-length biased input array, m-length array representation of label.
    '''
    return np.array([1] + input), decode(label, base=10)


def translateAll(data):
    yield from (translate(*datum) for datum in data)


if __name__ == '__main__':

    path_train = 'handwriting data/optdigits.tra'
    path_test = 'handwriting data/optdigits.tes'

    data_train = [translate(*datum) for datum in readData(path_train)]
    data_test = [translate(*datum) for datum in readData(path_test)]

    ann = Classifier(weights=randomWeights(64, 10))

    def train(neuralnet=None, data=data_train, learningRate=0.00001,
              convergenceThreshold=0.00001):
        if neuralnet is None:
            neuralnet = ann
        neuralnet.gradientDescent(data, learningRate, convergenceThreshold)

    def test(classifier=None, data=data_test):
        if classifier is None:
            classifier = ann
        score = classifier.numCorrect(data)
        num = len(data)
        print('{} of {} inputs correctly evaluated ({:0.3f})%.'.format(
              score, num, score / num * 100))

