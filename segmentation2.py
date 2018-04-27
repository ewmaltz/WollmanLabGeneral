import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage import filters
from skimage.morphology import h_maxima
from skimage.feature import peak_local_max
from scipy import ndimage
import classifyim
from skimage.filters import threshold_local, sobel, gaussian
from skimage.morphology import convex_hull_image


def segment_image(img, params=None):
    """
    ********************
    :param img:
    :param params:
    :return:
    """
    if params is None:
        params = {
                'cell_size_est': 0.17,  # ???
                'background_blur': 130,  # ??
                'image_blur': 1.5,  # gaussian kernel
                'block_size': 101,  # 2n-1 (1-inf)
                'thresh': 0.01,  # for binarization
                'smallest_object': 60,  # pixels
                'dist_intensity_ratio': 0.75,  # 0-1 weight
                'separation_distance': 8,  # pixels
                'edge_filter_blur': 2.0,  # kernel width in pixels
                'watershed_ratio': 0.15,  # 0-1 ratio of distance from edge vs bwgeodesic
                 }

    img_arr = np.array(img).astype(np.float)  # convert to numpy array
    img_norm = img_arr/np.max(img_arr)  # normalized img (0-1)
    img_pad = np.pad(img_norm, [1, 1], 'constant')  # pads edge with 0s
    imgSmooth = gaussian(img_pad, sigma=params['cell_size_est'])  # gaussian filter

    # Threshold
    img_hmax = h_maxima(imgSmooth, params['thresh'])
    img_hmax[np.where(img_hmax) == 1] = params['thresh']  # replace maxima with thresh value
    local_max_ixs = peak_local_max(img_hmax)
    RegionMax = img_hmax.copy()
    RegionMax[local_max_ixs[:, 0], local_max_ixs[:, 1]] = 1  # replace local maxima with 1s
    # RegionMax = filters.median(RegionMax)  # despeckle/clean array
    I = imgSmooth.copy()
    I[RegionMax] = 1  # set the ceiling to 1
    imgBW =


    # Generate Threshold
    # img_blur = threshold_local(img_arr, params['block_size'])  # functionally blurs image
    # img_bin = img_arr > img_blur  # binarize from threshold
    # img_thresh = filters.median(img_bin)  # despeckle binary image


    img_sobel = sobel(img_arr)  # sobel magnitude sqrt(sobel_h^2 + sobel_v^2)

    # plt.imshow(img_desp)
    # plt.show()

img = Image.open('test_cell_db.tif')
segged = segment_image(img)
#print(np.max(segged))


###### STEPS #######
"""
 1. gaussian filter
        >> gaussian(img_arr, sigma=params['cell_size_est'])
 2. imhmax on gaussian filter for thresholding
 ~  (1-2 might be accomplished with just threshold_local, binarize, and thresh)
 3. imregional_max on the thresholded image from 1-2
 4. clean up the maxed image
 5. Set ceiling to 1, max values of each region =1
 6. bwconvhull for voronoi cells
 7. distance transform
 8. watershed
 9. create voronoi cells by 
10. RegionBounds is 1 where watershed = 0
11. 
"""