import os

import numpy as np
from PIL import Image


def numLines(filepath):
    '''Find the number of lines in a file cheaply and quickly.'''
    # TODO: refactor into data-specific or util module.
    with open(filepath) as f:
        return sum(1 for line in f)


def readData(filepath):
    '''Read the data at the given file path, one line at a time.

    The data should be comma-seperated integer values, with the last number
    representing the label for the line of data.

    Args:
        filepath: The path of the data file to be opened.
    Yields:
        (data, label)
    '''
    # TODO: refactor into data-specific module.
    with open(filepath) as f:
        for line in f:
            numbers = [int(n) for n in line.strip().split(',')]
            yield numbers[:-1], numbers[-1]


def getImage(input):
    '''Generate an image visualizing the given data.

    Data is inverted and scaled from 0-255 for 8-bit grayscale, to look like
    black markings on a white background: inputs of 0 become white, and 16
    becomes (almost) black. (Inputs of 16 actually become 15 in the output, for
    even scaling.)

    Args:
        input (sequence of 64 values in the range [0, 16])
    Returns:
        8x8 Image representing the data.
    '''
    bitmap = np.array([(16 - n) * 15 + 15 for n in input], dtype=np.uint8)
    bitmap.resize(8, 8)
    return Image.fromarray(bitmap)


def makeImages(datapath, imgdir):
    '''Write PNG images of the data at datapath into imgdir.'''
    outputdir = os.path.join(imgdir, os.path.basename(datapath))
    os.makedirs(outputdir, exist_ok=True)
    # Find the number of digits for zero-padded one-indexed serial IDs.
    ID_len = len(str(numLines(datapath)))  # int(math.log10(numdata))+1? Pshh.
    for index, (input, label) in enumerate(readData(datapath)):
        ID = index + 1
        # str.format(**locals()) is dirty, but it's fine here.
        filename = '{ID:0{ID_len}d}_{label}.png'.format(**locals())
        getImage(input).save(os.path.join(outputdir, filename))


if __name__ == '__main__':
    makeImages('optdigits.tra', 'img')
    makeImages('optdigits.tes', 'img')
