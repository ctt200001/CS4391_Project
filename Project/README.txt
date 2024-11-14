The data_creater.py script goes through all the files in the original_images folder and creates reference images,
target images, and confidence images.

Reference images are used by the bilateral solver as a "ground truth" to smooth/filter the target image to. They are the
original images without any changes.

Target images are the reference images with gaussian noise introduced to them. These are the images being smoothed/filtered.

Confidence images display "confidence values" for each pixel in a target image. Confidence values measure how likely
a pixel is to be part of the original image (not noise). Lighter pixels (closer to white) denote a high confidence value,
while darker pixels (closer to black) denote a low confidence value.

When using any of the images in the bilateral solver, make sure their pixel values are normalized to values between [0, 1],
and change the type of the image to "np.float32" or "float32"  using "astype()". Afterwards, the image type should be changed
back to "uint8" and the pixel values should be multiplied by 255 so the pixel values are between [0, 255]
