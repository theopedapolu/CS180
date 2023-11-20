
# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import os
import skimage as sk
import skimage.io as skio

# Normalized Cross-Correlation
def NCC(A, B):
    avgA = np.average(A)
    stdA = np.std(A)
    avgB = np.average(B)
    stdB = np.std(B)

    dot_product = np.multiply(A - avgA, B - avgB) / (stdA * stdB)
    return np.average(dot_product)

# align image A with image B
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)
def search_window(A, B, window):
    A = sk.filters.sobel(A)
    B = sk.filters.sobel(B)
    # Crop 20% off the border of each image
    xcutoff = len(A) // 5
    ycutoff = len(A[0]) // 5
    croppedA = A[xcutoff:-xcutoff, ycutoff:-ycutoff]
    croppedB = B[xcutoff:-xcutoff, ycutoff:-ycutoff]

    best_displacement = [0,0]
    best_acc = float('-inf')
    # Find best displacement measurements
    for i in range(-window, window):
        for j in range(-window, window):
            displacedA = np.roll(np.roll(croppedA, i, 1), j, 0)
            score = NCC(displacedA, croppedB)
            if score > best_acc:
                best_acc = score
                best_displacement = [i,j]     
    return best_displacement                        
    
# Align images using a multi-scale image pyramid procedure by recursive scaling. Returns the best displacement
def align(A, B, depth):
    if depth == 0:
        best_displacement = search_window(A, B, 20)
    else:
        scaledA = sk.transform.rescale(A, 0.5)
        scaledB = sk.transform.rescale(B, 0.5)
        best_displacement = align(scaledA, scaledB, depth-1)
        best_displacement = [v*2 for v in best_displacement]

    best_i, best_j = best_displacement
    # Update estimate within a small window of [-3,3] pixels
    A = np.roll(np.roll(A, best_i, 1), best_j, 0)
    dither = search_window(A, B, 3)
    return [sum(x) for x in zip(best_displacement, dither)]

# Align image and return the shifted image after applying the best displacement
def getAlignedImage(A, B, depth):
    best_i, best_j = align(A, B, depth)
    resImage = np.roll(np.roll(A, best_i, 1), best_j, 0)
    return resImage, best_i, best_j

# Crops a single channel of an image
def cropChannel(A, threshold):
    whiteCutoff = 30/255
    A = sk.filters.sobel(A)
    top, left, bottom, right = 0, 0, len(A), len(A[0])

    # Get bottom border
    for i in range(len(A)//2, len(A)):
        num_white_bottom = 0
        for j in range(len(A[0])):
            if A[i][j] > whiteCutoff:
                num_white_bottom += 1     
        if num_white_bottom/len(A[0]) > threshold:
            bottom = i
            break

    # Get top border
    for i in range(len(A)//2, 0, -1):
        num_white_top = 0   
        for j in range(len(A[0])):        
            if A[i][j] > whiteCutoff:
                num_white_top += 1
        if num_white_top/len(A[0]) > threshold:
            top = i   
            break  

    # Get right border
    for j in range(len(A[0])//2, len(A[0])):
        num_white_right = 0
        for i in range(len(A)):
            if A[i][j] > whiteCutoff:
                num_white_right += 1        
        if num_white_right/len(A) > threshold:
            right = j 
            break      


    # Get left border
    for j in range(len(A[0])//2, 0, -1):
        num_white_left = 0
        for i in range(len(A)):
            if A[i][j] > whiteCutoff:
                num_white_left += 1
        if num_white_left/len(A) > threshold:
            left = j
            break

    return top, bottom, left, right

# Finds the cropping bounds for all 3 channels of the image and applies the maximum cropping overall. Returns the cropped image
def cropImage(A, B, C, threshold):
    t1, b1, l1, r1 = cropChannel(A, threshold)
    t2, b2, l2, r2 = cropChannel(B, threshold)
    t3, b3, l3, r3 = cropChannel(C, threshold)
    top, bottom, left, right = max(t1,t2,t3), min(b1,b2,b3), max(l1,l2,l3), min(r1,r2,r3)
    return A[top:bottom, left:right], B[top:bottom, left:right], C[top:bottom, left:right]

# Automatically constrasts the image and returns it
def autoContrast(A):
    intensityIm = A[:,:,:]
    for i in range(len(A)):
        for j in range(len(A[0])):
            intensityIm = (A[i][j][0] + A[i][j][1] + A[i][j][2])/3
    minval = np.percentile(A, 2)
    maxval = np.percentile(A, 98)
    image = np.clip(A, minval, maxval)
    image = ((image - minval)/(maxval-minval))
    return image     


# Make output directory if it doesn't exist
if not os.path.exists("outputs/"):
    os.mkdir("outputs/")

# Main Program
for imname in os.listdir("."):
    if imname.lower().endswith(".jpg") or imname.lower().endswith(".tif"):
        # set depth of image pyramid based on size of image
        depth = 1 if imname.lower().endswith(".jpg") else 4
        # Set threshold for cropping based on image size
        threshold = 0.7 if imname.lower().endswith(".jpg") else 0.3

        print("Aligning " + imname)
        # read in the image
        im = skio.imread(imname)

        # convert to double (might want to do this later on to save memory)    
        im = sk.img_as_float(im)
        # compute the height of each part (just 1/3 of total)
        height = np.floor(im.shape[0] / 3.0).astype(int)

        # separate color channels
        b = im[:height]
        g = im[height: 2*height]
        r = im[2*height: 3*height]

        # Pre-crop image
        b, g, r = cropImage(b, g, r, threshold)  

        # Align image and get displacement
        ag, best_i1, best_j1 = getAlignedImage(g, b, depth)
        ar, best_i2, best_j2 = getAlignedImage(r, b, depth)
        best_i = max(best_i1, best_i2, key = lambda x: abs(x))
        best_j = max(best_j1, best_j2, key = lambda x: abs(x))

        print("Red Displacement: ", best_i2, best_j2)
        print("Green Displacement: ", best_i1, best_j1)

        # Create a color image
        im_out = np.dstack([ar, ag, b])

        # Crop image further based on displacement
        if best_j < 0:
            im_out = im_out[0:best_j,:,:]
        else:
            im_out = im_out[best_j:,:,:]  

        if best_i < 0:
            im_out = im_out[:,0:best_i,:]
        else:
            im_out = im_out[:,best_i:,:]    

        # Auto-constrast image
        im_out = autoContrast(im_out)         

        # Convert floats back to ints
        im_out = sk.img_as_ubyte(im_out)

        # save the image
        fname = "outputs/out_" + imname[:-4] + ".jpg"
        skio.imsave(fname, im_out)

        # display the image
        #skio.imshow(im_out)
        #skio.show()
