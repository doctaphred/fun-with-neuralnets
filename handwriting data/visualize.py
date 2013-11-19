import os

import numpy as np
from PIL import Image


def numLines(filepath):
    '''Find the number of lines in a file cheaply and quickly.'''
    with open(filepath) as f:
        return sum(1 for line in f)


def readData(filepath):
    '''
    Yields:
        (data, label)
    '''
    with open(filepath) as f:
        for line in f:
            numbers = [int(n) for n in line.strip().split(',')]
            yield numbers[:-1], numbers[-1]


def getImage(input):
    # Invert and scale data from 0-255 for 8-bit grayscale.
    # Invert so it looks like black markings on a white background.
    # Actual range here is 15-255, for even scaling.
    bitmap = np.array([(16 - n) * 15 + 15 for n in input], dtype=np.uint8)
    bitmap.resize(8, 8)
    return Image.fromarray(bitmap)


def makeImages(datapath, imgdir):
    outputdir = os.path.join(imgdir, os.path.basename(datapath))
    os.makedirs(outputdir, exist_ok=True)
    # Find the number of digits required for zero-padded one-indexed serial IDs.
    ID_len = len(str(numLines(datapath))) # int(math.log10(numdata))+1? Pshh.
    for index, (input, label) in enumerate(readData(datapath)):
        ID = index + 1
        # str.format(**locals()) is dirty, but it's fine here.
        filename = '{ID:0{ID_len}d}_{label}.png'.format(**locals())
        getImage(input).save(os.path.join(outputdir, filename))


if __name__ == '__main__':
    makeImages('optdigits.tra', 'img')
    makeImages('optdigits.tes', 'img')
