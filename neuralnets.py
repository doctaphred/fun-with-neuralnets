import numpy as np
import random


def randomWeights(numInputs, numOutputs):
    return np.random.random((numInputs + 1, numOutputs))

def decode(number, base):
    output = np.zeros(base)
    output[number] = 1
    return output

class NeuralNet:

    def __init__(self, weights, learningRate):
        self.weights, self.learningRate = weights, learningRate

    def evaluate(self, input):
        '''
        input is a n-length numpy.array
        returns an n-length numpy array
        '''
        return np.dot(self.weights.T, input)

    def weightUpdates(self, input, output, target):
        '''Return the amounts to update the weights.

        input is a nx1 array.
        output and target are mx1 arrays.
        Returns an nxm array.
        '''
        return np.outer(-input, target - output)

    def gradientDescentStep(self, input, target):
        updates = self.weightUpdates(input, self.evaluate(input), target)
        self.weights += updates * self.learningRate

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
    inputs = []
    targets = []
    with open('/Users/frederick/Dropbox/data/handwriting/optdigits_100.tra') as f:
        for line in f:
            numbers = [int(n) for n in line.strip().split(',')]
            inputs.append(np.array([1] + numbers[:-1]))
            targets.append(np.array(decode(numbers[-1], base=10)))

    ann = NeuralNet(weights=randomWeights(64, 10), learningRate=0.00000001)

    def go():
        ann.gradientDescent(inputs, targets, convergenceThreshold=1)
