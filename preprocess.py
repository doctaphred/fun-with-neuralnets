import numpy as np


def normalize(data):
    '''Normalize a dataset.

    For each instance in the dataset, the mean vector is subtracted and each
        value is divided by its corresponding standard deviation. The resulting
        dataset has mean zero and variance one for each feature.

    Args:
        data ((n-length sequence of (m-length sequence)):
            A dataset consisting of n features and m instances.

    Returns:
        The normalized dataset.
    '''

    return ((data.T - np.mean(data, axis=1)) / np.std(data, axis=1)).T


def pca(data, k):
    '''Perform principal component analysis on a dataset.

    The top k eigenvectors are computed from the covariance matrix of the
    dataset. The dataset is then projected onto a k-dimensional subspace whose
    basis is the set of the top k eigenvectors.

    Args:
        data ((n-length sequence of (m-length sequence)):
            A dataset consisting of n features and m instances.
        k: integer
            The number of features in the transformed dataset.

    Returns:
        The dataset projected onto a k-dimensional subspace.
    '''

    mean = np.mean(data, axis=1)
    cov = np.cov(data)
    eigvals, eigvects = np.linalg.eig(cov)

    # Get the eigenvectors with the k largest eigenvalues
    idx = eigvals.argsort()[::-1][:k]
    transform = eigvects[:, idx]

    # Subtract the mean from the data
    dataZeroMean = (data.T - mean).T

    return transform.T.dot(dataZeroMean)
