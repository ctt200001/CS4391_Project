#This file takes all the images in original_images and makes corresponding reference images, target images,
#and confidence images

import cv2
import os
import numpy as np

#introduces noise to an image
def makeNoisyImage(image, mean, sigma):
    noise = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy_image = cv2.add(noise, image)
    return noisy_image

#makes confidence image from a noisy target image
def makeConfidenceImage(image, sigma=0.1):
    #convert target image into a greyscale float 32 image
    target = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    target = target / 255
    target = target.astype(np.float32)

    #compute gradients of target
    x_grad = cv2.Sobel(target, cv2.CV_32F, 1, 0, ksize=3)
    y_grad = cv2.Sobel(target, cv2.CV_32F, 0, 1, ksize=3)

    #calculate confidence values of each pixel
    grad_mag = np.sqrt(x_grad**2 + y_grad**2)

    #calculate the confidence image
    grad_mag = grad_mag / np.max(grad_mag)
    confidence_image = np.exp(-(grad_mag**2) / (2 * sigma**2))

    #multiply confidence values by 255 to make the image visible
    confidence_image = confidence_image * 255
    
    #convert confidence image back to uint8
    confidence_image = np.uint8(confidence_image)
    return confidence_image


#main
if __name__ == "__main__":
    #parameters for noise and confidence generation
    mean = 0
    sigma = 0.4

    for file in os.listdir("original_images"):
        #get file name and extension type
        ext_index = file.index('.')
        filename = file[:ext_index]
        ext_type = file[ext_index:]

        #read in image file
        image = cv2.imread("original_images/" + file)
        
        #save reference image to data folder
        cv2.imwrite("data/" + filename + "_reference" + ext_type, image)
        
        #create and save noisy image
        target_image = makeNoisyImage(image, mean, sigma)
        cv2.imwrite("data/" + filename + "_target" + ext_type, target_image)

        #create and save confidence image
        conf_image = makeConfidenceImage(target_image)
        cv2.imwrite("data/" + filename + "_confidence" + ext_type, conf_image)

        



        
        
