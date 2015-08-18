import os
import numpy as np
import scipy.io
import geometry_utils


class CUB_200_2011(object):
    NAME = 'CUB_200_2011'
    IMAGES_FOLDER_NAME = 'images'
    IMAGES_FILE_NAME = 'images.txt'
    TRAIN_TEST_SPLIT_FILE_NAME = 'train_test_split.txt'
    CLASS_LABEL_FILE_NAME = 'image_class_labels.txt'
    BBOX_FILE_NAME = 'bounding_boxes.txt'
    PARTS_FOLDER_NAME = 'parts'
    PARTS_FILE_NAME = 'parts.txt'
    PART_LOCS_FILE_NAME = 'part_locs.txt'
    SPLIT_FILE_TRAIN_INDICATOR = '1'
    SPLIT_FILE_TEST_INDICATOR = '0'

    def __init__(self, base_path, images_folder_name=None):
        self.base_path = base_path
        if images_folder_name:
            self.IMAGES_FOLDER_NAME = images_folder_name

        self.images_folder = os.path.join(
            self.base_path, self.IMAGES_FOLDER_NAME)
        self.images_file = os.path.join(
            self.base_path, self.IMAGES_FILE_NAME)
        self.train_test_split_file = os.path.join(
            self.base_path, self.TRAIN_TEST_SPLIT_FILE_NAME)
        self.class_label_file = os.path.join(
            self.base_path, self.CLASS_LABEL_FILE_NAME)
        self.bbox_file = os.path.join(
            self.base_path, self.BBOX_FILE_NAME)
        self.parts_file = os.path.join(
            self.base_path, self.PARTS_FOLDER_NAME, self.PARTS_FILE_NAME)
        self.part_locs_file = os.path.join(
            self.base_path, self.PARTS_FOLDER_NAME, self.PART_LOCS_FILE_NAME)

    def images(self):
        with open(self.images_file, 'r') as images_file:
            for line in images_file:
                parts = line.split()
                assert len(parts) == 2
                folder = self.images_folder
                yield {'img_id': parts[0],
                       'img_file': os.path.join(folder, parts[1]),
                       'img_file_rel': parts[1]}

    def image_addrs(self, relative=False):
        """
        returns a hash from image_id to image address, which can be relative to the images_folder name or absolute.
        By default it generates absolute addresses.
        """
        all_infos = list(self.images())
        the_hash = {}

        info_key = 'img_file'
        if relative:
            info_key = 'img_file_rel'

        for info in all_infos:
            the_hash[int(info['img_id'])] = info[info_key]

        return the_hash

    def train_test_id(self):
        """
        returns a tuple of 1d numpy arrays (IDtrain, IDtest), where each contains a list of img_ids for the corresponding set.
        """
        trains = []
        tests = []
        indicators = []
        with open(self.train_test_split_file, 'r') as split_file:
            for line in split_file:
                parts = line.split()
                assert len(parts) == 2
                img_id = parts[0]
                indicator = parts[1]
                indicators.append(indicator)
                if indicator == self.SPLIT_FILE_TRAIN_INDICATOR:
                    trains.append(img_id)
                elif indicator == self.SPLIT_FILE_TEST_INDICATOR:
                    tests.append(img_id)
                else:
                    raise Exception("Unknown indicator, %s" % indicator)

        len_trains = len(trains)
        len_tests = len(tests)
        IDtrain = np.zeros(len_trains, dtype=np.int)
        IDtest = np.zeros(len_tests, dtype=np.int)

        with open(self.class_label_file, 'r') as class_label:
            line_num = 0
            train_num = 0
            test_num = 0
            for line in class_label:
                parts = line.split()
                assert len(parts) == 2
                img_id = parts[0]
                indicator = indicators[line_num]
                if indicator == self.SPLIT_FILE_TRAIN_INDICATOR:
                    # training
                    IDtrain[train_num] = img_id
                    train_num += 1
                else:
                    # testing
                    IDtest[test_num] = img_id
                    test_num += 1

                line_num += 1

        return IDtrain, IDtest

    def bboxes(self):
        """
        generates a dictionary of image_id to bounding boxes
        """
        bbox_dict = {}
        bbox = np.genfromtxt(self.bbox_file, delimiter=' ')
        bbox = bbox[:, 1:]

        for i, b in enumerate(bbox):
            bbox_dict[i+1] = geometry_utils.Box(int(b[1]), int(b[1] + b[3]), int(b[0]), int(b[0] + b[2]))
        return bbox_dict

    def class_dict(self):
        """
        returns a dictionary from image_id to class_id.
        """
        class_dict = {}
        with open(self.class_label_file, 'r') as class_label:
            for line in class_label:
                parts = line.split()
                assert len(parts) == 2
                img_id = parts[0]
                img_cls = int(parts[1])
                class_dict[img_id] = img_cls

        return class_dict


class BerkeleyAnnotaionHelper(object):
    train_file_name = 'bird_train.mat'
    test_file_name = 'bird_test.mat'
    DEFAULT_BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'annotations')

    def __init__(self, cub, base_path=DEFAULT_BASE_PATH):
        self.base_path = base_path
        self.IDtrain, self.IDtest = cub.train_test_id()

        self.train_path = os.path.join(self.base_path, self.train_file_name)
        self.test_path = os.path.join(self.base_path, self.test_file_name)

        b_train_anno = scipy.io.loadmat(self.train_path)
        self.b_train_anno = b_train_anno['data']

        b_test_anno = scipy.io.loadmat(self.test_path)
        self.b_test_anno = b_test_anno['data']

    def get_train_berkeley_annotation(self, train_id, name):
        p = 0
        if name == 'head':
            p = 1
        elif name == 'body':
            p = 2
        elif name == 'bbox':
            p = 3
        res = self.b_train_anno[0, train_id][p][0]
        ymin, xmin, ymax, xmax = res[0], res[1], res[2], res[3]

        return geometry_utils.Box(xmin, xmax, ymin, ymax)

    def get_test_berkeley_annotation(self, test_id, name):
        p = 0
        if name == 'bbox':
            p = 1
        elif name == 'head':
            p = 2
        elif name == 'body':
            p = 3
        res = self.b_test_anno[0, test_id][p][0]
        ymin, xmin, ymax, xmax = res[0], res[1], res[2], res[3]

        return geometry_utils.Box(xmin, xmax, ymin, ymax)

    def annotation(self, img_id, name):
        train_where = np.argwhere(self.IDtrain == img_id)
        test_where = np.argwhere(self.IDtest == img_id)
        if train_where.shape[0] == 1:
            return self.get_train_berkeley_annotation(train_where[0, 0], name)
        elif test_where.shape[0] == 1:
            return self.get_test_berkeley_annotation(test_where[0, 0], name)
        else:
            raise Exception('Not found!')
