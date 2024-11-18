import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import collections as col

#To Do (i think):

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

#Computes the entire blur matrix multiplied by a vector using the equation in "Fast Bilateral-Space Stereo
#for Synthetic Defocus Supplemental Material"
#B.dot(x) = B1.dot(x) + B2.dot(x) + ... + BD.dot(x)
def computeBlur(blur_components, input):
    dp1 = blur_components[0].dot(input)
    dp2 = blur_components[1].dot(input)
    dp3 = blur_components[2].dot(input)
    dp4 = blur_components[3].dot(input)
    dp5 = blur_components[4].dot(input)

    return input + dp1 + dp2 + dp3 + dp4 + dp5

#Calculates the matrices to bistochastize W according to the algorithm in "The Fast Bilateral Solver"
def bistochastizeGrid(splat_matrix, blur_components):
    #mass of each grid cell (how many pixels are assigned to each grid cell) vector
    m = splat_matrix.dot(np.ones(splat_matrix.shape[1]))
    #n array
    n = np.ones(splat_matrix.shape[0])

    #do 10-20 iterations or until converged
    prev_n = n
    for i in range(15):
        print(i)
        n = np.sqrt((n * m) / computeBlur(blur_components, n))
        print(np.linalg.norm(n - prev_n))
        if(np.linalg.norm(n - prev_n) < 10e-6):
            break
        prev_n = n

    #create sparse matrices with the diagonals as n and m
    D_n = sp.diags(n)
    D_m = sp.diags(m)

    return D_n, D_m
        
            

#Return the bilateral grid of the argument image
#Implements the "grid creation" section of "real-time edge-aware image processing with the bilateral grid"
#expanded to 5 dimensions (the 6th dimension contains the cell's accumulated y, u, v, and count values)
#Also computes and returns the splat matrix, each dimension's blur matrix, and the slice matrix
#The size of the bilateral grid is dependent on the spatial sampling rate and the range sampling rate
#There are test statements that you can uncomment to visualize the matricies, they should look like what is
#presented on page 2 of the "Fast Bilateral-Space Stereo for Synthetic Defocus Supplemental Material"
def makeBilateralGrid(image, spatial_sampling_rate=8, range_sampling_rate=.2):
    #normalize image
    image = image.astype(np.float32)
    image = image / 255

    #get y, u, and v channels from image
    channel_Y, channel_U, channel_V = cv2.split(image)

    #calculate number of bins needed for bilateral grid for each channel
    bins_Y = int(np.max(channel_Y)/range_sampling_rate) + 1
    bins_U = int(np.max(channel_U)/range_sampling_rate) + 1
    bins_V = int(np.max(channel_V)/range_sampling_rate) + 1
    
    #calculate x and y dimensions of the intensity and count grids
    row_dim = round(image.shape[0]/spatial_sampling_rate)
    col_dim = round(image.shape[1]/spatial_sampling_rate)

    #create bilateral grid
    bilateral_grid = np.zeros([row_dim, col_dim, bins_Y, bins_U, bins_V, 4])

    #keep track of the number of pixels and vertices
    num_pixels = image.shape[0] * image.shape[1]
    num_vertices = 0
    
    #create hashmap of vertices (keys) and list of pixels corresponding to each vertex (value)
    vertices_and_pixels = col.defaultdict(list)

    #keep track of current pixel
    curr_pixel = 0

    #go through all pixels in image
    for image_y in range(image.shape[0]):
        for image_x in range(image.shape[1]):
            #calculate i, j, y, u, v values
            i = int(image_x/spatial_sampling_rate)
            j = int(image_y/spatial_sampling_rate)
            y = int(image[image_y, image_x, 0]/range_sampling_rate)
            u = int(image[image_y, image_x, 1]/range_sampling_rate)
            v = int(image[image_y, image_x, 2]/range_sampling_rate)

            #increment number of vertices if new one is being inserted into
            if(bilateral_grid[j, i, y, u, v, 3] == 0):
                num_vertices += 1

            #insert yuv and counter values into bilateral grid
            bilateral_grid[j, i, y, u, v, 0] += image[image_y, image_x, 0]
            bilateral_grid[j, i, y, u, v, 1] += image[image_y, image_x, 1]
            bilateral_grid[j, i, y, u, v, 2] += image[image_y, image_x, 2]
            bilateral_grid[j, i, y, u, v, 3] += 1

            #append vertex and pixel pair to hashmap
            #vertices_and_pixels[cell_index].append(curr_pixel)
            vertices_and_pixels[(j, i, y, u, v)].append(curr_pixel)

        
            #increment curr pixel
            curr_pixel += 1

    #create arrays for holding information from hashmap
    data = []
    rows = []
    cols = []

    #row index for current row of the sparse index, increments when adding new vertex
    sparse_row_index = 0
    #insert information in vertex and pixel hashmap into respective arrays to make sparse csr matrix
    for vertex, pixels in vertices_and_pixels.items():
        for pixel in pixels:
            data.append(1)
            rows.append(sparse_row_index)
            cols.append(pixel)
        sparse_row_index += 1
            
    
    #create splat matrix
    #the splat matrix is a sparse matrix with # of vertices rows and # of pixels columns that has a 1
    #for every vertex and pixel pair
    splat_matrix = sp.csr_matrix((data, (rows, cols)), shape=(num_vertices, num_pixels))
    
    #uncomment to see splat_matrix
    #plt.figure(figsize=(8, 8))
    #plt.spy(splat_matrix.toarray(), markersize=5)
    #plt.show()

    #calculate slice matrix here (transpose of the splat matrix)
    slice_matrix = splat_matrix.transpose()
    
    #uncomment to see slice matrix
    #plt.figure(figsize=(8, 8))
    #plt.spy(slice_matrix.toarray(), markersize=5)
    #plt.show()

    #create a constituent blur matrix for each dimension of the bilateral grid
    x_blur = sp.csr_matrix((num_vertices, num_vertices))
    y_blur = sp.csr_matrix((num_vertices, num_vertices))
    lum_blur = sp.csr_matrix((num_vertices, num_vertices))
    u_blur = sp.csr_matrix((num_vertices, num_vertices))
    v_blur = sp.csr_matrix((num_vertices, num_vertices))

    #list that holds all consituent blur matrices
    blur_components = [x_blur, y_blur, lum_blur, u_blur, v_blur]

    #go through all vertices in the hashmap and apply the [1 2 1] kernel
    for vertex, pixels in vertices_and_pixels.items():
        #apply kernel across all dimensions
        for curr_dim in range(5):
            
            #apply kernel to current vertex
            curr_index = list(vertices_and_pixels.keys()).index(vertex)
        
            #create arrays for making new csr_matrix, current vertex has weight of 2
            curr_data = [2] * len(pixels)
            curr_rows = [curr_index] * len(pixels)
            curr_cols = [curr_index] * len(pixels)
            
            # Add the central vertex's contribution to the current dimension blur matrix
            blur_components[curr_dim] = blur_components[curr_dim] + sp.csr_matrix((curr_data, (curr_rows, curr_cols)), shape=(num_vertices, num_vertices))

            #add or subtract 1 from the current dimension of the current vertex to get its left or right neighbor
            modified_vertex = list(vertex)  
            modified_vertex[curr_dim] -= 1
            left_neighbor = tuple(modified_vertex)
            modified_vertex[curr_dim] += 2
            right_neighbor = tuple(modified_vertex)
            
            #apply kernel to left neighbor if it exists
            if(left_neighbor in vertices_and_pixels):
                left_row_index = list(vertices_and_pixels.keys()).index(vertex)
                left_col_index = list(vertices_and_pixels.keys()).index(left_neighbor)
                
                #create arrays for creating new csr_matrix
                left_data = [1] * len(pixels)
                left_rows = [left_row_index] * len(pixels)
                left_cols = [left_col_index] * len(pixels)
                
                #add blur matrix of current vertex to corresponding dimension blur matrix
                blur_components[curr_dim] = blur_components[curr_dim] + sp.csr_matrix((left_data, (left_rows, left_cols)), shape=(num_vertices, num_vertices))
            
            #apply kernel to right neighbor if it exists
            if(right_neighbor in vertices_and_pixels):
                right_row_index = list(vertices_and_pixels.keys()).index(vertex)
                right_col_index = list(vertices_and_pixels.keys()).index(right_neighbor)
                
                #create arrays for creating new csr_matrix
                right_data = [1] * len(pixels)
                right_rows = [right_row_index] * len(pixels)
                right_cols = [right_col_index] * len(pixels)
                
                #add blur matrix of current vertex to corresponding dimension blur matrix
                blur_components[curr_dim] = blur_components[curr_dim] + sp.csr_matrix((right_data, (right_rows, right_cols)), shape=(num_vertices, num_vertices))

    #uncomment to see blur matrices
    plt.figure(figsize=(8, 8))
    #plt.spy(blur_components[0].toarray(), markersize=5)
    #plt.show()
    #plt.spy(blur_components[1].toarray(), markersize=5)
    #plt.show()
    #plt.spy(blur_components[2].toarray(), markersize=5)
    #plt.show()
    #plt.spy(blur_components[3].toarray(), markersize=5)
    #plt.show()
    #plt.spy(blur_components[4].toarray(), markersize=5)
    #plt.show()

    #return bilateral grid
    return bilateral_grid, slice_matrix, blur_components, splat_matrix

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

    #get bilateral grid, splat matrix, and slice matrix
    bilateral_grid, slice_matrix, blur_components, splat_matrix = makeBilateralGrid(yuv_reference)

    D_n, D_m = bistochastizeGrid(splat_matrix, blur_components)
    
