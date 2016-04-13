import settings
import sys
import os
import numpy as np
import cv2
import geometry_utils
sys.path.append(settings.CAFFE_PYTHON_PATH)
import caffe

DEFAULT_MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'models', 'caffenet_no_fc.prototxt')
IMAGENET_MEAN = np.array([104.00698793,  116.66876762,  122.67891434])


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
        """
        Extracts the features for a particular set of points.
        Call this function only after you have called `init_with_image`
        """
        n_points = points.shape[0]
        if layers is None:
            layers = self.layers
        n_features = self.ffeats.shape[2]
        features = np.zeros((n_points, n_features), dtype=np.float32)

        for i, point in enumerate(points):
            features[i, :] = self.ffeats[point[0], point[1], :]

        return features

    def image_point_features(self, img, part_box, part_name):
        """
        Extracts a set of positive and negative features from points generated from the image for a particular part.
        This function calls `init_with_image` so no need to call that yourself.
        """
        box = geometry_utils.Box.box_from_img(img)
        self.init_with_image(img)

        positive_points = part_box.generate_points_inside(param=settings.POISSON_PART_RADIUS[part_name], img=img)

        negative_points = box.generate_points_inside(param=settings.POISSON_NEGATIVE_RADIUS, img=img)
        negative_points = geometry_utils.filter_points(negative_points, part_box)

        return self.features(positive_points), self.features(negative_points)
