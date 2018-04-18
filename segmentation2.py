import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage import filters
from skimage.morphology import h_maxima
from skimage.feature import peak_local_max
from scipy import ndimage
import classifyim
from skimage.filters import threshold_local, sobel, gaussian


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
                'threshold': 0.25,  # for binarization
                'smallest_object': 60,  # pixels
                'dist_intensity_ratio': 0.75,  # 0-1 weight
                'separation_distance': 8,  # pixels
                'edge_filter_blur': 2.0,  # kernel width in pixels
                'watershed_ratio': 0.15,  # 0-1 ratio of distance from edge vs bwgeodesic
                 }

    img_arr = np.array(img).astype(np.float)  # convert to numpy array
    img_norm = img_arr/np.max(img_arr)  # normalized img (0-1)

    # Generate Threshold
    # img_blur = threshold_local(img_arr, params['block_size'])  # functionally blurs image
    # img_bin = img_arr > img_blur  # binarize from threshold
    # img_thresh = filters.median(img_bin)  # despeckle binary image


    img_sobel = sobel(img_arr)  # sobel magnitude sqrt(sobel_h^2 + sobel_v^2)

    # plt.imshow(img_desp)
    # plt.show()
    # img_gauss = gaussian(img_arr, sigma=params['cell_size_est'])  # smoothen

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