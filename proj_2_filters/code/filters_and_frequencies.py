import os
import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import skimage as sk
import skimage.io as skio
from align_image_code import align_images

def save_images(images, names):
    images = [sk.img_as_ubyte(im) for im in images]
    for image, name in zip(images, names):
        skio.imsave("outputs/" + name, image)

def save_cv_images(images, names):
    for image, name in zip(images, names):
        cv2.imwrite("outputs/" + name, image)

def finite_difference_operator(image, threshold):
    dx = np.array([[1,-1]])
    dy = dx.T
    image_dx = sp.signal.convolve2d(image, dx, mode="same")
    image_dy = sp.signal.convolve2d(image, dy, mode="same")
    image_mag = np.sqrt(np.square(image_dx) + np.square(image_dy))
    image_bin = (image_mag > threshold) * 255

    return image_dx, image_dy, image_mag, image_bin

def derivative_of_gaussian(image, threshold):
    # Apply gaussian kernel first then derivative operators
    G = cv2.getGaussianKernel(5,1)
    gaussian_kernel = G @ G.T
    image_blurred = sp.signal.convolve2d(image, gaussian_kernel, mode="same")
    image_dx, image_dy, image_mag, image_bin = finite_difference_operator(image_blurred, threshold)

    # Convolve kernels to get derivative of gaussian filters and apply them
    dx = np.array([[1,-1]])
    dy = dx.T
    DOG_dx = sp.signal.convolve2d(gaussian_kernel, dx)
    DOG_dy = sp.signal.convolve2d(gaussian_kernel, dy)
    image_DOG_dx = sp.signal.convolve2d(image, DOG_dx, mode="same")
    image_DOG_dy = sp.signal.convolve2d(image, DOG_dy, mode="same")
    image_DOG_mag = np.sqrt(np.square(image_DOG_dx) + np.square(image_DOG_dy))
    image_DOG_bin = (image_DOG_mag > threshold) * 255

    return DOG_dx, DOG_dy, image_blurred, image_dx, image_dy, image_mag, image_bin, image_DOG_dx, image_DOG_dy, image_DOG_mag, image_DOG_bin


def image_sharpening(image):
    # Derive the unsharp mask filter
    kernel_size = 5
    G = cv2.getGaussianKernel(kernel_size,1)
    gaussian_kernel = G @ G.T
    e = np.zeros((kernel_size, kernel_size))
    e[kernel_size//2][kernel_size//2] = 1
    unsharp_mask_kernel = 3*e - 2*gaussian_kernel

    sharpened_channels = []
    blurred_channels = []
    hi_freq_channels = []
    for i in range(3):
        channel = image[:,:,i]
        blurred_channels.append(sp.signal.convolve2d(channel, gaussian_kernel,mode="same"))
        hi_freq_channels.append(channel - blurred_channels[i])
        sharpened_channels.append(sp.signal.convolve2d(channel, unsharp_mask_kernel, mode="same"))

    blurred_image = np.dstack(blurred_channels)
    hi_freq_image = np.clip(np.dstack(hi_freq_channels),0,1)
    sharpened_image = np.clip(np.dstack(sharpened_channels),0,1)  
    return blurred_image, hi_freq_image, sharpened_image


def hybrid_channel(imageA, imageB, ksize1, ksize2, sigma1, sigma2):
    # Get low frequencies of imageA
    G1 = cv2.getGaussianKernel(ksize1,sigma1)
    gaussian_kernel1 = G1 @ G1.T
    low_pass_A = sp.signal.convolve2d(imageA, gaussian_kernel1, mode="same")

    # Get high frequencies of imageB
    G2 = cv2.getGaussianKernel(ksize2,sigma2)
    gaussian_kernel2 = G2 @ G2.T
    low_pass_B = sp.signal.convolve2d(imageB, gaussian_kernel2, mode="same")
    high_pass_B = imageB - low_pass_B

    # Average the images
    out_channel = (low_pass_A + high_pass_B) / 2
    return low_pass_A, high_pass_B, out_channel

def hybrid_images(imageA, imageB, ksize1, ksize2, sigma1, sigma2):
    # Align images
    imageA, imageB = align_images(imageA, imageB)
    # Convert back to uint (0-255)
    imageA = (imageA*255).astype(np.uint8)
    imageB = (imageB*255).astype(np.uint8)

    # Hybridize each channel
    out_channels = []
    low_pass_channels = []
    high_pass_channels = []
    for i in range(3):
        channelA = imageA[:,:,i]
        channelB = imageB[:,:,i]
        low, high, out  = hybrid_channel(channelA, channelB, ksize1, ksize2, sigma1, sigma2)
        low_pass_channels.append(low)
        high_pass_channels.append(high)
        out_channels.append(out)

    out_image = np.dstack(out_channels)
    low_image = np.dstack(low_pass_channels)
    high_image = np.dstack(high_pass_channels)
    return low_image, high_image, out_image  

def normalize_image(image):
    channels = []
    for i in range(3):
        channel = image[:,:,i]
        minimum = np.min(channel)
        maximum = np.max(channel)
        channels.append((channel - minimum) / (maximum - minimum))
    return np.dstack(channels)    


def gaussian_stack(image):
    g_stack = [image]
    for depth in range(5):
        G = cv2.getGaussianKernel(50, 2**depth)
        gaussian_kernel = G @ G.T
        channels = []
        for i in range(3):
            channel = image[:,:,i]
            blurred_channel = sp.signal.convolve2d(channel, gaussian_kernel, boundary="symm", mode="same")
            channels.append(blurred_channel)
        image = np.dstack(channels)
        g_stack.append(image)
    return g_stack

def laplacian_stack(image):
    g_stack = gaussian_stack(image)
    lp_stack = []
    for i in range(len(g_stack)-1):
        lp_level = g_stack[i] - g_stack[i+1]
        lp_stack.append(lp_level)
    lp_stack.append(g_stack[-1])  
    return lp_stack

def blend_images(imageA, imageB, mask):
    lp_A = laplacian_stack(imageA)
    lp_B = laplacian_stack(imageB)
    g_stack_mask = gaussian_stack(mask)
    A_images = []
    B_images = []
    blended_images = []
    collapse = np.zeros(lp_A[0].shape)
    for i in range(len(g_stack_mask)):
        level_A = lp_A[i]
        level_B = lp_B[i]
        level_mask = g_stack_mask[i]

        A_images.append(level_mask*level_A)
        B_images.append((1-level_mask)*level_B)
        new_image = level_mask*level_A + (1-level_mask)*level_B
        collapse = collapse + new_image
        blended_images.append(collapse)

    return A_images, B_images, blended_images  

def normalize_images(lst):
    return [normalize_image(im) for im in lst]

def get_fft(image):
    image = normalize_image(image)
    fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(image))))
    fft = normalize_images([fft])[0]
    fft = (fft*255).astype(np.uint8)
    return fft

def main():
    # Make output directory if it doesn't exist
    if not os.path.exists("outputs/"):
        os.mkdir("outputs/")

    # Part 1.1
    cameraman = cv2.imread("cameraman.png", cv2.IMREAD_GRAYSCALE)
    cameraman_derivatives = finite_difference_operator(cameraman, 50)
    cameraman_names = ["cameraman_dx.jpg", "cameraman_dy.jpg", "cameraman_mag.jpg", "cameraman_bin.jpg"]
    save_cv_images(cameraman_derivatives, cameraman_names)

    # #Part 1.2
    DOG_images = derivative_of_gaussian(cameraman, 20)
    DOG_names = ["DOG_dx.jpg","DOG_dy.jpg","blurred_cameraman.jpg","cameraman_blurred_dx.jpg","cameraman_blurred_dy.jpg","cameraman_blurred_mag.jpg","cameraman_blurred_bin.jpg",
                 "cameraman_DOG_dx.jpg","cameraman_DOG_dy.jpg","cameraman_DOG_mag.jpg","cameraman_DOG_bin.jpg"]
    save_cv_images(DOG_images, DOG_names)

    # Part 2.1
    # taj = skio.imread("taj.jpg")
    # taj = sk.img_as_float(taj)
    # taj_ims = image_sharpening(taj)
    # save_images(taj_ims, ["blurred_taj.jpg","hi_freq_taj.jpg","sharpened_taj.jpg"])

    # angel = skio.imread("angel.jpg")
    # angel = sk.img_as_float(angel)
    # angel_ims = image_sharpening(angel)
    # save_images(angel_ims, ["blurred_angel.jpg","hi_freq_angel.jpg","sharpened_angel.jpg"])

    # lion = skio.imread("lion.jpg")
    # lion = sk.img_as_float(lion)

    # G = cv2.getGaussianKernel(7,3)
    # gaussian_kernel = G @ G.T
    # lion_blurred_channels = []
    # for i in range(3):
    #     channel = lion[:,:,i]
    #     lion_blurred_channels.append(sp.signal.convolve2d(channel, gaussian_kernel, mode="same"))
    # lion_blurred = np.dstack(lion_blurred_channels)    
    # save_images([lion_blurred], ["blurred_lion.jpg"])

    # lion_blurred_ims = image_sharpening(lion_blurred)
    # save_images(lion_blurred_ims, ["double_blurred_lion.jpg","hi_freq_lion.jpg","sharpened_lion.jpg"])

    # Part 2.2
    # Convert read images to float (0-1) for aligning operations
    # derek = (cv2.imread("DerekPicture.jpg", cv2.IMREAD_COLOR)/255).astype(np.float32)
    # nutmeg = (cv2.imread("Nutmeg.jpg", cv2.IMREAD_COLOR)/255).astype(np.float32)
    # hybrid_dutmeg = hybrid_images(derek, nutmeg, 20, 20, 5, 10)
    # fft_dutmeg = normalize_images([hybrid_dutmeg])[0]
    # fft_dutmeg = np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid_dutmeg))))
    # save_cv_images([hybrid_dutmeg, fft_dutmeg], ["dutmeg.jpg", "fft_dutmeg.jpg"])



    # happy = (cv2.imread("happy_elon.jpg", cv2.IMREAD_COLOR)/255).astype(np.float32)
    # smoking = (cv2.imread("smoking_elon.jpg", cv2.IMREAD_COLOR)/255).astype(np.float32)
    # hybrid_elon = hybrid_images(smoking, happy, 30, 20, 3, 5)
    # save_images([hybrid_elon], ["hybrid_elon.jpg"])

    # tiger = (cv2.imread("tiger.jpg", cv2.IMREAD_COLOR)/255).astype(np.float32)
    # lawrence = (cv2.imread("lawrence.jpg", cv2.IMREAD_COLOR)/255).astype(np.float32)
    # low, high, hybrid_tigrence = hybrid_images(lawrence, tiger, 30, 20, 3, 10)

    # fft_tiger = get_fft(tiger)
    # fft_lawrence = get_fft(lawrence)
    # fft_low = get_fft(low)
    # fft_high = get_fft(high)
    # fft_tigrence = get_fft(hybrid_tigrence)
    # hybrid_tigrence = (hybrid_tigrence*255).astype(np.uint8)
    # save_cv_images([hybrid_tigrence], ["hybrid_tigrence.jpg"])

    # bus = (cv2.imread("bus.jpg", cv2.IMREAD_COLOR)/255).astype(np.float32)
    # giraffe = (cv2.imread("giraffe.jpg", cv2.IMREAD_COLOR)/255).astype(np.float32)
    # low, high, biraffe  = hybrid_images(giraffe, bus, 30, 20, 3, 10)
    # save_cv_images([biraffe], ["hybrid_biraffe.jpg"])


    # zebra = (cv2.imread("zebra.jpg", cv2.IMREAD_COLOR)/255).astype(np.float32)
    # cow = (cv2.imread("cow.jpg", cv2.IMREAD_COLOR)/255).astype(np.float32)
    # hybrid_cowra = hybrid_images(zebra, cow, 30, 20, 3, 10)
    # save_cv_images([hybrid_cowra], ["hybrid_cowra.jpg"])


    # Part 2.3 & 2.4
    # apple = skio.imread("apple.jpeg")
    # apple = sk.img_as_float(apple)
    # orange = skio.imread("orange.jpeg")
    # orange = sk.img_as_float(orange)
    # mask = np.ones(apple.shape)
    # mask[:,(apple.shape[1]//2):, :] = 0.0

    # apple_ims, orange_ims, oraple_ims = blend_images(apple, orange, mask)
    # oraple = np.clip(oraple_ims[-1], 0, 1)
    # apple_ims = normalize_images(apple_ims)
    # orange_ims = normalize_images(orange_ims)
    # oraple_ims = normalize_images(oraple_ims)
    # save_images(apple_ims, ["lp_apple0.jpg","lp_apple1.jpg","lp_apple2.jpg","lp_apple3.jpg","lp_apple4.jpg", "lp_apple5.jpg"])
    # save_images(orange_ims, ["lp_orange0.jpg","lp_orange1.jpg","lp_orange2.jpg","lp_orange3.jpg","lp_orange4.jpg", "lp_orange5.jpg"])
    # save_images(oraple_ims, ["lp_oraple0.jpg","lp_oraple1.jpg","lp_oraple2.jpg","lp_oraple3.jpg","lp_oraple4.jpg", "lp_oraple5.jpg"])

    # save_images([mask, oraple], ["apple_mask.jpg","oraple.jpg"])

    # toothless = skio.imread("toothless.jpeg")
    # toothless = sk.img_as_float(toothless)
    # gate = skio.imread("golden_gate.jpg")
    # gate = sk.img_as_float(gate)
    # toothless_mask = skio.imread("toothless_mask.jpeg")
    # toothless_mask = np.dstack([toothless_mask, toothless_mask, toothless_mask])
    # toothless_mask = sk.img_as_float(toothless_mask)

    # toothless_ims, gate_ims, toothless_gate_ims = blend_images(toothless, gate, toothless_mask)
    # toothless_gate = np.clip(toothless_gate_ims[-1], 0, 1)
    # toothless_ims = normalize_images(toothless_ims)
    # gate_ims = normalize_images(gate_ims)
    # toothless_gate_ims = normalize_images(toothless_gate_ims)

    # save_images(toothless_ims, ["lp_toothless0.jpg","lp_toothless1.jpg","lp_toothless2.jpg","lp_toothless3.jpg","lp_toothless4.jpg", "lp_toothless5.jpg"])
    # save_images(gate_ims, ["lp_gate0.jpg","lp_gate1.jpg","lp_gate2.jpg","lp_gate3.jpg","lp_gate4.jpg", "lp_gate5.jpg"])
    # save_images(toothless_gate_ims, ["lp_toothless_gate0.jpg","lp_toothless_gate1.jpg","lp_toothless_gate2.jpg","lp_toothless_gate3.jpg","lp_toothless_gate4.jpg", "lp_toothless_gate5.jpg"])
    # save_images([toothless_gate], ["toothless_gate.jpg"])

    # bill = skio.imread("bill.jpg")
    # bill = sk.img_as_float(bill)
    # sather = skio.imread("sather.jpg")
    # sather = sk.img_as_float(sather)
    # bill_mask = skio.imread("bill_mask.jpg")
    # bill_mask = np.dstack([bill_mask, bill_mask, bill_mask])
    # bill_mask = sk.img_as_float(bill_mask)

    # bill_ims, sather_ims, bill_sather_ims = blend_images(bill, sather, bill_mask)
    # bill_sather = np.clip(bill_sather_ims[-1], 0, 1)
    # bill_ims = normalize_images(bill_ims)
    # sather_ims = normalize_images(sather_ims)
    # bill_sather_ims = normalize_images(bill_sather_ims)

    # save_images(bill_ims, ["lp_bill0.jpg","lp_bill1.jpg","lp_bill2.jpg","lp_bill3.jpg","lp_bill4.jpg", "lp_bill5.jpg"])
    # save_images(sather_ims, ["lp_sather0.jpg","lp_sather1.jpg","lp_sather2.jpg","lp_sather3.jpg","lp_sather4.jpg", "lp_sather5.jpg"])
    # save_images(bill_sather_ims, ["lp_bill_sather0.jpg","lp_bill_sather1.jpg","lp_bill_sather2.jpg","lp_bill_sather3.jpg","lp_bill_sather4.jpg", "lp_bill_sather5.jpg"])
    # save_images([bill_sather], ["bill_sather.jpg"])

main()