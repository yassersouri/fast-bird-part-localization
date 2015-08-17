import os
import numpy as np
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

    def get_all_images(self):
        with open(self.images_file, 'r') as images_file:
            for line in images_file:
                parts = line.split()
                assert len(parts) == 2
                folder = self.images_folder
                yield {'img_id': parts[0],
                       'img_file': os.path.join(folder, parts[1]),
                       'img_file_rel': parts[1]}

    def get_all_image_addrs(self, relative=False):
        """
        returns a hash from image_id to image address, which can be relative to the images_folder name or absolute.
        By default it generates absolute addresses.
        """
        all_infos = list(self.get_all_images())
        the_hash = {}

        info_key = 'img_file'
        if relative:
            info_key = 'img_file_rel'

        for info in all_infos:
            the_hash[int(info['img_id'])] = info[info_key]

        return the_hash

    def get_train_test_id(self):
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

    def get_bbox(self):
        """
        generates a set of bounding_boxes
        """
        bbox_dict = {}
        bbox = np.genfromtxt(self.bbox_file, delimiter=' ')
        bbox = bbox[:, 1:]

        for i, b in enumerate(bbox):
            bbox_dict[i+1] = geometry_utils.Box(int(b[0]), int(b[1]), int(b[0]) + int(b[3]), int(b[0]) + int(b[2]))
        return bbox_dict

    def get_class_dict(self):
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
