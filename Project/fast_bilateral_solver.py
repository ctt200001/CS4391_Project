import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

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




#Return the bilateral grid of the argument image
#Implements the "grid creation" section of "real-time edge-aware image processing with the bilateral grid"
#expanded to 5 dimensions (the 6th dimension contains the cell's accumulated y, u, v, and count values)

def makeBilateralGrid(image, spatial_sampling_rate=8, range_sampling_rate=8):
    #normalize image
    image = image.astype(np.float32)
    image = image / 255
    
    #calculate x and y dimensions of the intensity and count grids
    row_dim = round(image.shape[0]/spatial_sampling_rate)
    col_dim = round(image.shape[1]/spatial_sampling_rate)

    #create bilateral grid
    bilateral_grid = np.zeros([row_dim, col_dim, range_sampling_rate, range_sampling_rate, range_sampling_rate, 4])

    #create splat matrix (short, wide, sparse)
    num_pixels = image.shape[0] * image.shape[1]
    num_cells = row_dim * col_dim * (range_sampling_rate**3)
    splat_matrix = sp.lil_matrix((num_pixels, num_cells))
    
    #get sizes of each dimension of bilateral grid for use when inserting into splat matrix
    x_shape = bilateral_grid.shape[0]
    y_shape = bilateral_grid.shape[1]
    lum_shape = bilateral_grid.shape[2]
    u_shape = bilateral_grid.shape[3]

    #keep track of current pixel
    curr_pixel = 0

    #go through all pixels in image
    for image_y in range(image.shape[0]):
        for image_x in range(image.shape[1]):
            #calculate i and j values
            i = int(image_x/spatial_sampling_rate)
            j = int(image_y/spatial_sampling_rate)
            #only use luminance of curr pixel to calculate k for now
            y = int(image[image_y, image_x, 0]/range_sampling_rate)
            u = int(image[image_y, image_x, 1]/range_sampling_rate)
            v = int(image[image_y, image_x, 2]/range_sampling_rate)

            #insert yuv and counter values into bilateral grid
            bilateral_grid[j, i, y, u, v, 0] += image[image_y, image_x, 0]
            bilateral_grid[j, i, y, u, v, 1] += image[image_y, image_x, 1]
            bilateral_grid[j, i, y, u, v, 2] += image[image_y, image_x, 2]
            bilateral_grid[j, i, y, u, v, 3] += 1

            #calculate column index of splat matrix to insert into (the 1d equivalent index of the grid cell we
            #we just inserted into)
            cell_index = (v * x_shape * y_shape * lum_shape * u_shape) + (u * x_shape * y_shape * lum_shape) + (y * x_shape * y_shape) + (j * y_shape) + i 
            #change the corresponding cell in the splat matrix to 1
            splat_matrix[curr_pixel, cell_index] = 1
        
            #increment curr pixel
            curr_pixel += 1
    
    splat_matrix = splat_matrix.tocsr()
    print(splat_matrix)    

    #calculate blur matrix here (small and sparse)

    #calculate slice matrix here (transpose of the splat matrix)

    #maybe add bistochastization algorithm here?
    

    #return bilateral grid
    return bilateral_grid
    
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

    #get bilateral grid
    bilateral_grid = makeBilateralGrid(yuv_reference)
    
