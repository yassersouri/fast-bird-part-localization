import settings
import sys
import os
import numpy as np
import cv2
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
