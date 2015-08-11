import settings
import sys
import os
import numpy as np
import cv2
from poisson_disk import PoissonDiskSampler
sys.path.append(settings.CAFFE_PYTHON_PATH)
import caffe

DEFAULT_MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'models', 'caffenet_no_fc.prototxt')
IMAGENET_MEAN = np.array([104.00698793,  116.66876762,  122.67891434])


class Box(object):
    """
    This class represents a box in an image. This could be a bounding box of an object or part.
    Internally each box is represented by a tuple of 4 integers: (xmin, xmax, ymin, ymax)
    """

    POINT_GENERATION_POLECIES = ['poisson_disk']

    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    @staticmethod
    def box_from_cendim(cen, dim):
        """
        Create a box from a pair of center and dimension. Each center or dimension is a tuple. For short we call the center and dimension the `cendim`
        Center: (cenX, cenY)
        Dimension: (height, width)
        """
        cenX, cenY = cen
        height, width = dim
        height_2 = height / 2.
        width_2 = width / 2.
        xmin = int(round(cenX - height_2))
        xmax = int(round(cenX + height_2))
        ymin = int(round(cenY - width_2))
        ymax = int(round(cenY + width_2))
        return Box(xmin, xmax, ymin, ymax)

    def cendim(self):
        """
        Convert the box into cendim format. In cendim format the center and dimension are stored as floating point numbers.
        """
        cenX = float(self.xmin + self.xmax) / 2
        cenY = float(self.ymin + self.ymax) / 2
        height = float(self.xmax - self.xmin)
        width = float(self.ymax - self.ymin)

        cen = (cenX, cenY)
        dim = (height, width)
        return cen, dim

    def trim_to_borders(self, img):
        """
        Trims the box with respect to the image provided.
        """
        img_h, img_w = img.shape[:2]
        self.xmin = max(0, self.xmin)
        self.xmax = min(img_h - 1, self.xmax)
        self.ymin = max(0, self.ymin)
        self.ymax = min(img_w - 1, self.ymax)

        return self

    def draw_box(self, img, color=(1, 0, 0), width=2):
        """
        Annotate the `img` with this Box. This returns a new image with the box annotated on it.
        """
        new_img = img.copy()

        cv2.rectangle(new_img, (self.ymin, self.xmin), (self.ymax, self.xmax), color, width)
        return new_img

    def get_sub_image(self, img):
        """
        Return a sub-image only containing information inside this Box.
        """
        self.trim_to_borders(img)

        return img[self.xmin:self.xmax, self.ymin:self.ymax]

    @staticmethod
    def expand_cendim(cen, dim, alpha):
        height, width = dim

        height = (2 * alpha) * height
        width = (2 * alpha) * width

        dim = (height, width)
        return cen, dim

    def expand(self, alpha=0.666):
        cen, dim = self.cendim()
        cen, dim = Box.expand_cendim(cen, dim, alpha)
        new_box = Box.box_from_cendim(cen, dim)
        self.xmin = new_box.xmin
        self.xmax = new_box.xmax
        self.ymin = new_box.ymin
        self.ymax = new_box.ymax

        return self

    def evalIOU(self, get_box, source_shape):
        # TODO
        # making sure not to generate errors further down the line
        self.trim_to_borders(source_shape)
        get_box.trim_to_borders(source_shape)

        height, width = source_shape[:2]

        gt_part = np.zeros((height, width), np.uint8)
        gt_part[get_box.xmin:get_box.xmax, get_box.ymin:get_box.ymax] = 1

        sl_part = np.zeros((height, width), np.uint8)
        sl_part[self.xmin:self.xmax, self.ymin:self.ymax] = 1

        intersection = (gt_part & sl_part).sum()
        union = (gt_part | sl_part).sum()

        return intersection / float(union)

    def evalPCP(self, get_box, source_shape, thresh=0.5):
        iou = self.evalIOU(get_box, source_shape)
        if iou >= thresh:
            return 1
        else:
            return 0

    def generate_points_inside(self, policy='poisson_disk', param=None, img=None):
        """
        This function generates points inside this rectangle. It uses the poisson disk to do it by default. But there is a policy option that is configurable.
        There is an optional `param` parameter that specifies the parameters of the generation policy.

        Different Policies:
            - `poisson_disk`:
                The param is expected to be the radius. The radius is the parameter of the poisson disk sampler.
                By default radius is set to be average of 1/10 of width and height of the box.

        Each point is a row vector [x, y]. A set of `n` points will be represented as a numpy array of shape (n,2). The dtype is numpy.int.

        There can be an optional img option. We can use the image's shape to further prune points that are located outside the boundary of the image.
        """
        assert(policy in self.POINT_GENERATION_POLECIES)
        cen, dim = self.cendim()
        height, width = dim
        if policy == 'poisson_disk':
            if param is None:
                radius = ((height / 10.) + (width / 10.)) / 2.
            else:
                radius = param
            pds = PoissonDiskSampler(width, height, radius)
            samples = pds.get_sample()
            points = np.zeros((len(samples), 2), dtype=np.int)
            for i, s in enumerate(samples):
                points[i, :] = [int(round(s[0])), int(round(s[1]))]

            points += np.array([self.xmin, self.ymin])

        return points


def draw_points(points, ax, color=None):
    if color is None:
        color = 'red'
    for p in points:
        ax.plot(p[0], p[1], 'o', color=color)


class DeepHelper(object):

    @staticmethod
    def get_caffenet(gpu_mode=True):
        """
        returns a net and a transformer for that net in a tuple: (net, transformer)
        """
        net = caffe.Net(DEFAULT_MODEL_FILE, settings.CAFFE_NET_PRETRAINED, caffe.TEST)

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', IMAGENET_MEAN)
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))
        if gpu_mode:
            caffe.set_mode_gpu()
        return net, transformer

    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    feats = {}
    num_feats = {}

    def __init__(self, net_layers=None, interpolation=cv2.INTER_LINEAR):
        self.net, self.transformer = self.get_caffenet(settings.GPU_MODE)
        if net_layers is not None:
            self.layers = net_layers

        self.interpolation_type = interpolation

    def init_with_image(self, img, half=False):
        # resizing transformer and network with respect to the input image
        self.input_image_size = img.shape[:2]

        if half:
            self.input_image_size = (self.input_image_size[0] / 2, self.input_image_size[1] / 2)

        new_size = [1, 3, self.input_image_size[0], self.input_image_size[1]]
        self.transformer.inputs['data'] = new_size
        self.net.blobs['data'].reshape(*new_size)

        # preprocessing the image
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', img)

        # forward pass
        self.net.forward()

        self._make_features_ready()

    def _make_features_ready(self):
        for layer in self.layers:
            data = self.net.blobs[layer].data[0]

            data = data.swapaxes(0, 2)
            data = data.swapaxes(0, 1)

            # since the opencv's size is (cols, rows) and not (height, width) as in numpy
            data = cv2.resize(data, (self.input_image_size[1], self.input_image_size[0]), interpolation=self.interpolation_type)

            _, _, num_feat = data.shape

            self.num_feats[layer] = num_feat
            self.feats[layer] = data

        self.ffeats = np.concatenate([self.feats[k] for k in self.layers], axis=2)

    def features(self, points, layers=None):
        n_points = len(points)
        if layers is None:
            layers = self.layers
        n_features = sum(self.num_feats[l] for l in layers)
        features = np.zeros((n_points, n_features))

        for i, point in enumerate(points):
            # @TODO fix this.
            x, y = point.y - 1, point.x - 1  # not because I'm idoit, but because of other things!
            # feat_layers = [self.feats[l][x, y, :] for l in layers]
            features[i, :] = self.ffeats[x, y, :]

        return features
