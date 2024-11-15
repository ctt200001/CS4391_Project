import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#To Do (i think):
#1. The bilateral grid needs to be bistochastized. This should be done in another function or in the
#   makeBilateralGrid function. "The Fast Bilateral Solver" paper has more information on the algorithm
#   needed to do that (algorithm 1 on page 18), but I believe there is another algorithm as well.

#2. Implement functions to construct the A matrix and b vector in equation 6 of the fast bilateral solver paper
#   so that we can solve the linear system in equation 8.

#3. A function where we can backproject the "solved" bilateral grid onto a pixel space, which results in
#   our final image.

#4. Set up some of the things mentioned in the "Preconditioning & Initialization" section of the fast
#   bilateral solver paper? I don't think these are actually necessary, but they improve our run time.
#   Probably save this for if we have time.

#5. Functions to test our implementation of the fast bilateral solver. Maybe we can compare the PSNR/MSE 
#   between the original image and our output vs. the original image and the output of a bilateral filter
#   to see how well our implementation smooths/filters out noise. Additionally, we could also measure
#   the amount of time it takes our implementation to run vs how long it takes a bilateral filter or another
#   bilateral solver to run to see if ours is "fast"?




#Returns two 3 dimensional grids
#spatal_sampling_rate = downsampling factor that we divide image dimensions by to get bilateral grid dimensions
#range_sampling_rate = number of bins per channel
#Implements the "grid creation" section of the "real-time edge-aware image processing with the bilateral grid"
#paper except the image intensity values and count values are split into two separate 3d grids for ease of use.
#They can be combined by making a 3d grid of arrays and storing corresponding intensities and values in the
#same cell.
#Use both grids in tandem to have the bilateral grid.
def makeBilateralGrid(image, spatial_sampling_rate=4, range_sampling_rate=8):
    #calculate x and y dimensions of the intensity and count grids
    row_dim = round(image.shape[0]/spatial_sampling_rate)
    col_dim = round(image.shape[1]/spatial_sampling_rate)

    #create bilateral grid
    intensity_grid = np.zeros([row_dim, col_dim, range_sampling_rate])
    count_grid = np.zeros([row_dim, col_dim, range_sampling_rate])

    #normalize image
    image = image.astype(np.float32)
    image = image / 255

    #go through all pixels in image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            #calculate i, j, and k values
            i = int(x/spatial_sampling_rate)
            j = int(y/spatial_sampling_rate)
            #only use luminance of curr pixel to calculate k for now
            k = int(image[y, x, 0]/range_sampling_rate)

            #trying to insert luminance (y) of curr pixel in the image intensity dimension for now
            #might need to average u and v values later
            intensity_grid[j, i, k] = intensity_grid[j, i, k] + image[y, x, 0]
            #increment corresponding cell in count_grid
            count_grid[j, i, k] = count_grid[j, i, k] + 1
    
    #uncomment the three lines below for testing
    #plt.imshow(intensity_grid[:, :, 0])
    #plt.colorbar()
    #plt.show()


    #maybe add bistochastization algorithm here?


    #return grids
    return intensity_grid, count_grid
    
#Converts a RGB image to a YUV image
#Returns YUV image in normalized float 32 nparray
#Probably not needed since we can use cv2 to convert to YUV
def RGBToYUV(image):
    #normalize image
    image = image.astype(np.float32)
    image = image / 255
    
    #flatten image for dot product with yuv_matrix
    original_shape = image.shape
    flattened_image = image.reshape(-1, 3)

    #yuv matrix
    yuv_matrix = np.array([[0.299, 0.587, 0.114],
                  [-0.147, -0.289, 0.436],
                  [0.615, -0.515, -0.100]])
    
    #calculate the yuv values by computing the dot product
    yuv_image = np.dot(flattened_image, np.transpose(yuv_matrix))
    #reshape yuv_image back to original dimensions
    yuv_image = yuv_image.reshape(original_shape)
    
    return yuv_image
    

#Main
if __name__ == "__main__":
    #change these variables to match what image you want to work with
    filename = "cablecar"
    extension = ".bmp"
    
    #read in data images for corresponding image name
    reference = np.copy(cv2.imread("data/" + filename + "_reference" + extension))
    target = np.copy(cv2.imread("data/" + filename + "_target" + extension))
    confidence = np.copy(cv2.imread("data/" + filename + "_confidence" + extension))

    #convert reference image to YUV colorspace
    yuv_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2YUV)

    #get the intensity grid and count grid
    instensity_grid, count_grid = makeBilateralGrid(yuv_reference)
