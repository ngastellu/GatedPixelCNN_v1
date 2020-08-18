import numpy as np

def sample_augment(image, xdim, ydim, normalize, binarize, rotflip, n_samples): # if brigthness = 0 it will not be normalized, if 1 it will be, if 2 it will be binarized about the median
    samples = np.zeros((n_samples, xdim, ydim))
    if normalize == 1: # batch-normalize brightness
        avg_brightness = np.average(image)
    else:
        avg_brightness = 1

    # potential top-left pixels to choose from
    pot_samples = image[:-ydim, :-xdim]  # can't take any samples which go outside of the image!
    y_rands = np.random.randint(0, image.shape[0]-ydim, n_samples)
    x_rands = np.random.randint(0, image.shape[1]-xdim, n_samples)
    if rotflip == 1:
        rot_rands = np.random.randint(0, 4, n_samples)
        flip_rands = np.random.randint(0, 2, n_samples)
    else:
        rot_rands = np.zeros(n_samples)
        flip_rands = np.zeros(n_samples)

    for i in range(n_samples):
        slice = image[y_rands[i]:y_rands[i]+ydim, x_rands[i]:x_rands[i]+xdim] # grab a random sample from the image
        # orient it randomly
        if flip_rands[i]:
            slice = np.fliplr(slice)

        slice = np.rot90(slice, rot_rands[i])


        if normalize == 1:  #normalize brightness
            slice = slice * avg_brightness // np.average(slice) # correct to the average
            samples[i, :, :] = np.array(slice).astype('uint8')
        elif binarize == 1:  #binarize by the median
            slice = slice > np.median(slice)
            samples[i, :, :] = np.array(slice).astype('bool')
        else: # leave as-is
            samples[i, :, :] = np.array(slice)

    return samples

