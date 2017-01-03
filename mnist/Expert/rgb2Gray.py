import numpy as np
##import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread('input.png')     
gray = rgb2gray(img)
gray.save('output.png')
##plt.imshow(gray, cmap = plt.get_cmap('gray'))
##plt.show()
