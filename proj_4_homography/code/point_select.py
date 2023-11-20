import matplotlib.pyplot as plt
import skimage as sk
import skimage.io as skio
import numpy as np

def get_points(im):
    plt.imshow(im)
    pts = plt.ginput(-1, timeout=0)
    pts = [np.ceil(np.array(p[::-1])).astype(int) for p in pts]
    print("\nCoordinates of all selected points: ")
    print(np.array(pts))

# vlsb_left = skio.imread("vlsb_left.jpg")
# vlsb_middle =skio.imread("vlsb_middle.jpg")
# vlsb_right = skio.imread("vlsb_right.jpg")
# DL_sign = skio.imread("DL_sign.jpg")
# park_left = skio.imread("park_left.jpg")
# park_middle = skio.imread("park_middle.jpg")
# park_right = skio.imread("park_right.jpg")
#book = skio.imread("images/book.jpg")
#painting = skio.imread("images/painting.jpg")
# DL_left = skio.imread("DL_left.jpg")
# DL_right = skio.imread("DL_right.jpg")
# campanile_left = skio.imread("campanile_left.jpg")
# campanile_middle = skio.imread("campanile_middle.jpg")
sproul_left = skio.imread("sproul_left.jpg")
sproul_right = skio.imread("sproul_right.jpg")

# get_points(vlsb_left)
# get_points(vlsb_middle)

# get_points(park_left)
# get_points(park_middle)
# get_points(book)
#get_points(book)
#get_points(painting)
# get_points(painting)

# get_points(campanile_left)
# get_points(campanile_middle)
get_points(sproul_left)
get_points(sproul_right)

