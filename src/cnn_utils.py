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

    def init_with_image(self, img):
        # resizing transformer and network with respect to the input image
        self.input_image_size = img.shape[:2]
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
            x, y = point.y - 1, point.x - 1  # not because I'm idoit, but because of other things!
            # feat_layers = [self.feats[l][x, y, :] for l in layers]
            features[i, :] = self.ffeats[x, y, :]

        return features

    def part_for_image(self, all_image_infos, all_segmentaion_infos, cub_parts, img_id, part_filter_names, N_part=10, N_bg=100):
        img = caffe.io.load_image(all_image_infos[img_id])
        seg = thresh_segment_mean(caffe.io.load_image(all_segmentaion_infos[img_id]))

        self.init_with_image(img)

        parts = cub_parts.for_image(img_id)
        part_parts = parts.filter_by_name(part_filter_names)
        part_positive = gen_part_points(part_parts.get_rect_info(img.shape), seg, N_part)
        part_negative = gen_bg_points(part_parts.get_rect_info(img.shape), seg, N_bg)

        # TODO: we don't have input_dim any more we have an input_image_size
        part_positive.norm_for_size(img.shape[1], img.shape[0], self.input_dim)
        part_negative.norm_for_size(img.shape[1], img.shape[0], self.input_dim)

        feats_positive = self.features(part_positive)
        feats_negative = self.features(part_negative)

        return feats_positive, feats_negative

    def part_for_image_local(self, all_image_infos, all_segmentaion_infos, bah, img_id, part_name, N_part, N_bg):
        img = caffe.io.load_image(all_image_infos[img_id])
        seg = thresh_segment_mean(caffe.io.load_image(all_segmentaion_infos[img_id]))

        self.init_with_image(img)

        part_rect_info = bah.get_berkeley_annotation(img_id, part_name)
        part_positive = gen_part_points(part_rect_info, seg, N_part)
        part_negative = gen_bg_points(part_rect_info, seg, N_bg)

        # TODO: we don't have input_dim any more we have an input_image_size!
        part_positive.norm_for_size(img.shape[1], img.shape[0], self.input_dim)
        part_negative.norm_for_size(img.shape[1], img.shape[0], self.input_dim)

        feats_positive = self.features(part_positive)
        feats_negative = self.features(part_negative)

        return feats_positive, feats_negative

    def part_features_for_rf(self, all_image_infos, all_segmentaion_infos, cub_parts, IDs, part_filter_names, N_part=10, N_bg=100):
        positives = []
        negatives = []
        for i, img_id in enumerate(IDs):
            feats_positive, feats_negative = self.part_for_image(all_image_infos, all_segmentaion_infos, cub_parts, img_id, part_filter_names, N_part, N_bg)

            positives.append(feats_positive)
            negatives.append(feats_negative)
        X_pos = np.vstack(positives)
        y_pos = np.ones((X_pos.shape[0]), np.int)
        X_neg = np.vstack(negatives)
        y_neg = np.zeros((X_neg.shape[0]), np.int)

        X = np.vstack((X_pos, X_neg))
        y = np.concatenate((y_pos, y_neg))

        return X, y

    def part_features_for_local_rf(self, all_image_infos, all_segmentaion_infos, bah, IDs, part_name, N_part=10, N_bg=100):
        positives = []
        negatives = []

        for i, img_id in enumerate(IDs):
            feats_positive, feats_negative = self.part_for_image_local(all_image_infos, all_segmentaion_infos, bah, img_id, part_name, N_part, N_bg)

            positives.append(feats_positive)
            negatives.append(feats_negative)

        X_pos = np.vstack(positives)
        y_pos = np.ones((X_pos.shape[0]), np.int)
        X_neg = np.vstack(negatives)
        y_neg = np.zeros((X_neg.shape[0]), np.int)

        X = np.vstack((X_pos, X_neg))
        y = np.concatenate((y_pos, y_neg))

        return X, y
