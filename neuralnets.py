import numpy as np
import random


def randomWeights(numInputs, numOutputs):
    return np.random.random((numInputs + 1, numOutputs)) / 1000

def decode(number, base):
    output = np.zeros(base)
    output[number] = 1
    return output

class NeuralNet:
    '''Single-layer neural network.

    Attributes:
        weights (nxm array): n inputs, m outputs
        learningRate
    '''

    def __init__(self, weights, learningRate):
        self.weights, self.learningRate = weights, learningRate

    def evaluate(self, input):
        '''
        Args: input (n-length array)
        Returns: an n-length array.
        '''
        return np.dot(self.weights.T, input)

    def weightUpdates(self, input, output, target):
        '''Return the amounts to update the weights.

        Args:
            input (n-length array): input to evaluate.
            output, target (m-length arrays): actual and desired outputs.

        Returns:
            an nxm array.
        '''
        return np.outer(input, target - output)

    def gradientDescentStep(self, input, target):
        updates = self.weightUpdates(input, self.evaluate(input), target)
        self.weights += updates * self.learningRate

    def updateWeights(self, input, target):
        '''
        dw_ij = a * (t_j - y_j) * g_(h_j) * x_i
        where
            a = learning rate
            t_j = target output
            y_j = actual output
            g_ = derivative of activation function
            h_j = weighted sum of inputs (g_(h_j) = 1 for perceptrons)
            x_i = ith input
        '''
        # TODO: ???
        self.weights += np.outer(input, target - output) * self.learningRate

    def averageError(self, inputs, targets):
        return sum(sum(abs(self.evaluate(i) - t)) for i, t in zip(inputs, targets)) / len(inputs)

    def gradientDescent(self, inputs, targets, convergenceThreshold):
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
                self.gradientDescentStep(input, target)


if __name__ == '__main__':

    datapath_training = '/Users/frederick/Dropbox/data/handwriting/optdigits_100.tra'

    inputs = []
    targets = []
    with open(datapath_training) as f:
        for line in f:
            numbers = [int(n) for n in line.strip().split(',')]
            inputs.append(np.array([1] + numbers[:-1]))
            targets.append(np.array(decode(numbers[-1], base=10)))

    ann = NeuralNet(weights=randomWeights(64, 10), learningRate=0.00001)

    def train(threshold=0.0001):
        ann.gradientDescent(inputs, targets, convergenceThreshold=threshold)

    def eval(number=0):
        return ann.evaluate(inputs[number])

    def check(number=0):
        return ann.evaluate(inputs[number]).argmax() == targets[number].argmax()

    def checkall():
        return sum(check(number) for number in range(len(inputs)))
