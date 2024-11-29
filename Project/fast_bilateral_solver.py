import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import collections as col

#4. Set up some of the things mentioned in the "Preconditioning & Initialization" section of the fast
#   bilateral solver paper? I don't think these are actually necessary, but they improve our run time.
#   Probably save this for if we have time.

#5. Functions to test our implementation of the fast bilateral solver. Maybe we can compare the PSNR/MSE 
#   between the original image and our output vs. the original image and the output of a bilateral filter
#   to see how well our implementation smooths/filters out noise. Additionally, we could also measure
#   the amount of time it takes our implementation to run vs how long it takes a bilateral filter or another
#   bilateral solver to run to see if ours is "fast"?




#Fills in the U and V chanels of the output yuv image with the average values inside the bilateral_grid.
#Each pixel corresponds to a vertex (a grid cell that at least one pixel is assigned to).
#Get the coordinates of that grid cell using the pixels and vertices hashmap, get the accumulated U and V
#values there, and average them out by the counter (keeps track of the number of pixels assigned to that vertex)
#to get the average values
def getColors(output_yuv, bilateral_grid, pixels_and_vertices):
    curr_pixel = 0
    #loop through all pixels in output_yuv
    for output_y in range(output_yuv.shape[0]):
        for output_x in range(output_yuv.shape[1]):
            #calculate the coordinates of the vertex in the bilateral grid
            coords = pixels_and_vertices[curr_pixel]
            j = coords[0]
            i = coords[1]
            y = coords[2]
            u = coords[3]
            v = coords[4]

            #calculate the average u and v values
            avg_u = bilateral_grid[j, i, y, u, v, 1] / bilateral_grid[j, i, y, u, v, 3]
            avg_v = bilateral_grid[j, i, y, u, v, 2] / bilateral_grid[j, i, y, u, v, 3]

            #assign the average u and v values to the second and third channel
            output_yuv[output_y, output_x, 1] = avg_u
            output_yuv[output_y, output_x, 2] = avg_v

            #increment current pixel
            curr_pixel +=1
    
    return output_yuv

#Calculates A and b from eq. 6 so that we can solve eq. 7 (Ay = b) from the fast bilateral solver paper
def prepareEquationParameters(splat_matrix, blur_components, target, confidence, D_n, D_m, smoothing_lambda=64):
    #normalize target image
    target = target.astype(np.float32)
    target = target / 255

    #normalize confidence image
    confidence = confidence.astype(np.float32)
    confidence = confidence / 255
    
    #calculate a and b
    a = smoothing_lambda * (D_m - (D_n.dot(computeBlur(blur_components, D_n)))) + (sp.diags(splat_matrix.dot(confidence.flatten())))
    b = splat_matrix.dot((confidence * target).flatten())
    return a, b

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
        n = np.sqrt((n * m) / computeBlur(blur_components, n))
        #break from loop if converged
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
#Also computes and returns the splat matrix, each dimension's blur matrix, and the slice matrix.
#The splat matrix maps vertex to pixel correspondance, and contains a 1 at indices where pixels are assigned
#to the vertex.
#The blur matrix is calculated by computing the dot product of a vector with each dimension's blur matrix.
#To get each dimensions blur matrix, you need to apply the [1 2 1] kernel to each dimension, so for all the
#vertices in the grid, the function checks to see if it has neighbors and computes the kernel on them as well
#if they do.
#The slice matrix is the inverse of the splat.
#The size of the bilateral grid is dependent on the the sampling rates.
#There are test statements that you can uncomment to visualize the matricies, they should look like what is
#presented on page 2 of the "Fast Bilateral-Space Stereo for Synthetic Defocus Supplemental Material"
def makeBilateralGrid(image, spatial_sampling_rate=5.5, lum_sampling_rate=.12, uv_sampling_rate=.08):
    #normalize image
    image = image.astype(np.float32)
    image = image / 255

    #get y, u, and v channels from image
    channel_Y, channel_U, channel_V = cv2.split(image)

    #calculate number of bins needed for bilateral grid for each channel
    bins_Y = int(np.max(channel_Y)/lum_sampling_rate) + 1
    bins_U = int(np.max(channel_U)/uv_sampling_rate) + 1
    bins_V = int(np.max(channel_V)/uv_sampling_rate) + 1
    
    #calculate x and y dimensions of the intensity and count grids
    row_dim = int(image.shape[0]/spatial_sampling_rate) + 1
    col_dim = int(image.shape[1]/spatial_sampling_rate) + 1

    #create bilateral grid
    bilateral_grid = np.zeros([row_dim, col_dim, bins_Y, bins_U, bins_V, 4])

    #keep track of the number of pixels and vertices
    num_pixels = image.shape[0] * image.shape[1]
    num_vertices = 0
    
    #create hashmaps for later use
    vertices_and_pixels = col.defaultdict(list)
    vertices_and_indices = col.defaultdict(list)
    pixels_and_vertices = col.defaultdict(list)

    #keep track of current pixel
    curr_pixel = 0

    #go through all pixels in image
    for image_y in range(image.shape[0]):
        for image_x in range(image.shape[1]):
            #calculate i, j, y, u, v values
            i = int(image_x/spatial_sampling_rate)
            j = int(image_y/spatial_sampling_rate)
            y = int(image[image_y, image_x, 0]/lum_sampling_rate)
            u = int(image[image_y, image_x, 1]/uv_sampling_rate)
            v = int(image[image_y, image_x, 2]/uv_sampling_rate)

            #increment number of vertices if new one is being inserted into
            if(bilateral_grid[j, i, y, u, v, 3] == 0):
                vertices_and_indices[(j, i, y, u, v)] = num_vertices
                num_vertices += 1

            #store curr pixel and coordinate pair
            pixels_and_vertices[curr_pixel] = [j, i, y, u, v]

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
    #lt.figure(figsize=(8, 8))
    #plt.spy(splat_matrix, markersize=5)
    #plt.show()

    #calculate slice matrix here (transpose of the splat matrix)
    slice_matrix = splat_matrix.transpose()
    
    #uncomment to see slice matrix
    #plt.figure(figsize=(8, 8))
    #plt.spy(slice_matrix, markersize=5)
    #plt.show()

    #create a constituent blur matrix for each dimension of the bilateral grid
    x_blur = sp.csr_matrix((num_vertices, num_vertices))
    y_blur = sp.csr_matrix((num_vertices, num_vertices))
    lum_blur = sp.csr_matrix((num_vertices, num_vertices))
    u_blur = sp.csr_matrix((num_vertices, num_vertices))
    v_blur = sp.csr_matrix((num_vertices, num_vertices))

    #list that holds all consituent blur matrices
    blur_components = [x_blur, y_blur, lum_blur, u_blur, v_blur]

    #apply kernel across all dimensions
    for curr_dim in range(5):
        #holds weights for each dimension
        data = []
        rows = []
        cols = []

        #go through all vertices and apply the [1 2 1] kernel
        for vertex, pixels in vertices_and_pixels.items():
            #current vertex has a weight of 2
            curr_index = vertices_and_indices[vertex]
            data.extend([2] * len(pixels))  
            rows.extend([curr_index] * len(pixels))  
            cols.extend([curr_index] * len(pixels)) 
            
            #get coordinates of left and right neighbors
            modified_vertex = list(vertex)
            modified_vertex[curr_dim] -= 1
            left_neighbor = tuple(modified_vertex)
            modified_vertex[curr_dim] += 2
            right_neighbor = tuple(modified_vertex)

            #apply kernel to left neighbor if it exists
            if(left_neighbor in vertices_and_pixels):
                left_col_index = vertices_and_indices[left_neighbor]
                data.extend([1] * len(pixels))
                rows.extend([curr_index] * len(pixels))
                cols.extend([left_col_index] * len(pixels))
                
            #apply kernel to right neighbor if it exists
            if(right_neighbor in vertices_and_pixels):
                right_col_index = vertices_and_indices[right_neighbor]
                data.extend([1] * len(pixels))
                rows.extend([curr_index] * len(pixels))
                cols.extend([right_col_index] * len(pixels))
        
        #add dimension blur matrix to blur components
        blur_components[curr_dim] += sp.csr_matrix((data, (rows, cols)), shape=(num_vertices, num_vertices))

    #uncomment to see blur matrices
    #plt.figure(figsize=(8, 8))
    #plt.spy(blur_components[0], markersize=5)
    #plt.show()
    #plt.spy(blur_components[1], markersize=5)
    #plt.show()
    #plt.spy(blur_components[2], markersize=5)
    #plt.show()
    #plt.spy(blur_components[3], markersize=5)
    #plt.show()
    #plt.spy(blur_components[4], markersize=5)
    #plt.show()

    return bilateral_grid, slice_matrix, blur_components, splat_matrix, pixels_and_vertices

#Main
if __name__ == "__main__":
    #change these variables to match what image you want to work with
    filename = "yacht"
    extension = ".bmp"
    
    #read in data images for corresponding image name
    reference = np.copy(cv2.imread("data/" + filename + "_reference" + extension))
    target = np.copy(cv2.imread("data/" + filename + "_target" + extension, cv2.IMREAD_GRAYSCALE))
    target_rgb = np.copy(cv2.imread("data/" + filename + "_target" + extension))
    confidence = np.copy(cv2.imread("data/" + filename + "_confidence" + extension, cv2.IMREAD_GRAYSCALE))

    #convert reference image to YUV colorspace
    yuv_reference = cv2.cvtColor(reference, cv2.COLOR_RGB2BGR)
    yuv_reference = cv2.cvtColor(yuv_reference, cv2.COLOR_BGR2YUV)

    #get bilateral grid, splat matrix, slice matrix, and pixels and vertices hashmap
    bilateral_grid, slice_matrix, blur_components, splat_matrix, pixels_and_vertices = makeBilateralGrid(yuv_reference)

    #calculate bistochastize matrices
    D_n, D_m = bistochastizeGrid(splat_matrix, blur_components)

    #get A and b
    a, b = prepareEquationParameters(splat_matrix, blur_components, target, confidence, D_n, D_m)
    
    #add regularization to a's diagonal to avoid matrix singularization
    reg_lam = 10e-6
    a = a + reg_lam * sp.csr_matrix(np.eye(a.shape[0]))

    #solve for y, then compute dot product with slice matrix (more efficient than getting transpose of a)
    y = sp.linalg.spsolve(a, b)
    output = slice_matrix.dot(y)

    #reshape output to matrix
    output = output.reshape((reference.shape[0], reference.shape[1]))

    #reshape output to have 3 channels
    output_yuv = np.zeros((output.shape[0], output.shape[1], 3))
    #put the grayscale output image in the first channel (this replaces the Y value)
    output_yuv[..., 0] = output

    #fill the rest of the channels and rescale YUV values in output
    output_yuv = getColors(output_yuv, bilateral_grid, pixels_and_vertices)
    output_yuv = output_yuv * 255
    output_yuv = np.uint8(output_yuv)

    #convert the yuv image back to rgb
    output_rgb = cv2.cvtColor(output_yuv, cv2.COLOR_YUV2RGB)
    
    #get a version filtered by the opencv bilateral filter for comparison
    cv2_filtered = cv2.bilateralFilter(target_rgb, d=9, sigmaColor = 75, sigmaSpace = 75)

    #write both to the output folder
    cv2.imwrite("output/" + filename + "_output" + extension, output_rgb)
    cv2.imwrite("output/" + filename + "_cv2" + extension, cv2_filtered)

    print("Filtering Completed")
