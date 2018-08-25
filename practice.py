import numpy as np
from matplotlib import pyplot as plt

npz_path = "C:\\Users\\Eldan Chodorov\\"  # Todo: complete this!
npz_path_roi = "C:\\Users\\Eldan Chodorov\\"  # Todo: complete this!


def show_npz(path):

    data = np.loadz(path)  # dict of arrays

    # if keyError occurs, first print data.keys()
    first = data["arr_0"]
    second = data["arr_1"]
    third = data["arr_2"]

    show_img(first)
    show_img(second)
    show_img(third)

def show_img(array):
    plt.figure()
    plt.imshow(array)
    plt.show()

if __name__ == '__main__':
    # should show 6 images
    show_npz(npz_path)

