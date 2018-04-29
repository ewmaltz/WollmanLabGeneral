import numpy as np
from skimage.measure import regionprops
from segmentation import segment_image
import cv2


def get_nuc_cyto_ratios(db_imgs, marker_imgs, cyto_size, params):
    """
    Take nuclear-stained images, calculate the region props to find each cell nucleus,
    then dilate around each nucleus to find the cytoplasmic area. Map this back onto the
    original image and record the ratio of the average intensities of each area.
    :param db_images: DeepBlue images for segmentation etc.
    :param cyto_size:
    :param params: dict for segmentation params
    :return:
    """
    nuc_cyto_ratios = []
    for i in range(len(db_imgs)):
        print(i, 'out of ', len(db_imgs))
        seg_img = segment_image(db_imgs[i], params)
        regions = regionprops(seg_img)
        for region in regions:
            # create blank matrix to hold each nuclear region in the image
            nuc = np.zeros(seg_img.shape)
            nuc[list(region.coords.T)] = 1

            # dilate to get cytoplasmic area
            kernel = np.ones((5, 5), np.uint8)
            cyto = cv2.dilate(nuc, kernel, iterations=cyto_size)

            # get ratios
            avg_cyto = np.mean(marker_imgs[i][cyto])
            avg_nuc = np.mean(marker_imgs[i][nuc])
            nuc_cyto_ratios.append(avg_nuc / avg_cyto)

    return nuc_cyto_ratios


def colorize_segmented_image(img, color_type):
    """
    Returns a randomly colorized segmented image for display purposes.
    :param img: Should be a numpy array of dtype np.int and 2D shape
    :param color_type: 'rg' for red green gradient, 'rb' = red blue, 'bg' = blue green
    :return: Randomly colorized, segmented image (shape=(n,m,3))
    """
    # get empty rgb_img as a skeleton to add colors
    rgb_img = np.zeros((img.shape[0], img.shape[1], 3))

    # make your colors
    num_cells = np.max(img)  # find the number of cells so you know how many colors to make
    colors = np.random.random_integers(0, 255, (num_cells, 3))
    if 'r' in color_type:
        if 'g' in color_type:
            colors[:, 2] = 0  # remove blue
        else:
            colors[:, 1] = 0  # remove green
    else:
        colors[:, 0] = 0  # remove red

    regions = regionprops(img)
    for i in range(1, len(regions)):  # start at 1 because no need to replace background (0s already)
        rgb_img[list(regions[i].coords.T)] = colors[i]  # won't use the 1st color

    return rgb_img.astype(np.int)
