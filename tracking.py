import numpy as np
from segmentation import segment_image
import matplotlib.pyplot as plt

# Placeholder for tracking framework
def optimize_params(img, params, epochs):
    for i in range(epochs):
        for key in params.keys():
            if key != 'smallest_object':
                keys = []
                params[key] *= 1.01
                keys.append(params[key])
                seg_img = segment_image(img, params)
                params[key] *= .99
                keys.append(params[key])
                seg_img2 = segment_image(img, params)
                counts = [np.max(seg_img), np.max(seg_img2)]
                params[key] = keys[counts.index(max(counts))]
                print(i, max(counts))
        print(params)
        plt.imshow(seg_img2)
        plt.show()


