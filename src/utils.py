import matplotlib.pyplot as plt
import numpy as np
import torchvision


def show_example_images(dataloader, classes):
    # get some random images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'%5s' % classes[labels[j]] for j in range(4)))


def imshow(img):
    '''
    :param img: image normalize to [-1, 1]
    :return:
    '''
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
