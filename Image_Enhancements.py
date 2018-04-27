import numpy as np
import dit
from dit.multivariate import entropy
from scipy.ndimage.filters import gaussian_filter as gf


def auto_bright_nonlin(img, epochs, transform_factor=0.5, sigma=0.8, mean_thresh=2, mean_reduction=0.9):
    """
    TODO: Transform multiple images simultaneously (e.g. Before and After) per Roy's request

    Try sliding window approach and maximize entropy for each window. Windows can't be too small, or too large.
    :param img: numpy array/obj of the image you want to transform
    :param epochs: hyperparameter for number of transformations
    :param transform_factor: hyperparameter for rate of exponential transformation
    :param sigma: gaussian filter hyperparameter
    :param mean_thresh: hyperparameter controlling sensitivity of intensity cutoff
    :param mean_reduction: hyperparameter for reducing the lowest intensity pixels
    :return best_img: maximum entropy image
    """
    # normalize pixels between 0 and 1
    img = np.array(img).astype(np.float)
    img *= 1/np.max(img)

    # calculate initial entropy of the image
    counts, bins = np.histogram(img)
    count_frac = [count / np.sum(counts) for count in counts]
    d = dit.Distribution(list(map(str, range(len(counts)))), count_frac)
    entropy_loss = [entropy(d)]
    d_entropy = 1  # arbitrary
    imgs = [img]  # holds all images so that we can choose the one with the best entropy
    for i in range(epochs):
        # remove low intensity pixels
        img[img <= mean_thresh*np.mean(img)] *= mean_reduction
        img = gf(img, sigma=sigma)
        img = img ** (1-(transform_factor*d_entropy))
        img[img == np.inf] = 1  # clip infities at 1
        imgs.append(img)
        counts, bins = np.histogram(img)
        count_frac = [count / np.sum(counts) for count in counts]
        d = dit.Distribution(list(map(str, range(len(counts)))), count_frac)
        entropy_loss.append(entropy(d))
        d_entropy = entropy_loss[-1] - entropy_loss[-2]
        if i % 10 == 0:
            print('Finished: ', 100 * i / epochs, '%')

    print('Best entropy: ', max(entropy_loss), 'at ix ', entropy_loss.index(max(entropy_loss)))
    best_img = imgs[entropy_loss.index(max(entropy_loss))]
    best_img = gf(best_img, sigma=sigma)
    return best_img, entropy_loss


def auto_bright_lin():
    pass