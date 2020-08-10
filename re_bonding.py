import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_aa, line
import tqdm

# import npy structures

#run = 'model=2_dataset=11_dataset_size=20000_filters=64_layers=4_filter_size=7_noise=0.0_denvar=0.0_T=0.000'
#images = np.load('samples/' + run +'.npy', allow_pickle=True)
#images = images == np.amax(images)


def re_bond(images):
    #images = np.load(path,allow_pickle=True)

    images = images * 2

    max_bond_length = int(1.8 / 0.2) # max bond length is about 1.5A, grid is about 0.2 A
    empty = np.zeros((images.shape[0], images.shape[-2] + max_bond_length * 2, images.shape[-1] + max_bond_length * 2))
    empty[:, max_bond_length:-max_bond_length, max_bond_length:-max_bond_length] = images
    images = empty
    bonded = images * 1

    def draw_line(image, n, yi, xi, yf, xf):
        #rr, cc, val = line_aa(yi, xi, yf, xf)
        rr, cc = line(yi, xi, yf, xf)
        image[n, rr, cc] = 1#val
        return image

    # search each image
    if np.average(images) < 0.2:
        for n in range(images.shape[0]):
            for i in range(images[n, :, :].shape[-2]):
                for j in range(images[n, :, :].shape[-1]):
                    if images[n, i, j] != 0:  # if we find a particle
                        neighbors = 0
                        radius = []
                        neighborx = []
                        neighbory = []
                        for ii in range(i - max_bond_length, i + max_bond_length + 1):
                            for jj in range(j - max_bond_length, j + max_bond_length + 1):  # search in a ring around it of radius (max bond length)
                                if images[n, ii, jj] != 0:
                                    if not ((i == ii) and (j == jj)) : # if we find a particle that is not the original one, store it's location
                                        rad = (np.sqrt((i - ii) ** 2 + (j - jj) ** 2))
                                        if rad <= max_bond_length:
                                            radius.append(rad)
                                            neighborx.append(jj)
                                            neighbory.append(ii)
                                            neighbors += 1

                        # draw lines!!
                        for r in range(neighbors):
                            bonded = draw_line(bonded, n, i, j, neighbory[r], neighborx[r])
    else:
        bonded = bonded * 1


    bonded = (bonded > 0).astype('uint8')
    bonded += images.astype('uint8') > 0
    bonded = bonded[:, max_bond_length:-max_bond_length, max_bond_length:-max_bond_length]
    #plt.imshow(bonded[0,:,:])
    #np.save('bonded_MAC', bonded)

    return bonded
