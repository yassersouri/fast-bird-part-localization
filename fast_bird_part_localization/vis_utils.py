import matplotlib.pylab as plt
import cv2
import skimage


def vis(preds_prob, img, ax=None, fig=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

    preds_prob_resize = cv2.resize(preds_prob, (img.shape[1], img.shape[0]))
    img_gray = skimage.color.rgb2gray(img)

    cax = ax.matshow(preds_prob_resize, cmap=plt.cm.Reds, alpha=1)
    ax.imshow(img_gray, alpha=0.3, cmap=plt.cm.gray)

    fig.colorbar(cax)
    ax.axis('off')
