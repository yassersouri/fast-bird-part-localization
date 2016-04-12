import sys
from datetime import datetime as dt
sys.path.append('.')
from fast_bird_part_localization import settings
from fast_bird_part_localization import cnn_utils
from fast_bird_part_localization import dataset_utils
import numpy as np
import sklearn.ensemble
import sklearn.metrics
from sklearn.externals import joblib
sys.path.append(settings.CAFFE_PYTHON_PATH)
import caffe


def main():
    cub = dataset_utils.CUB_200_2011(settings.CUB_FOLDER_PATH)
    imgs_addr = cub.image_addrs()
    imgs_anno = dataset_utils.BerkeleyAnnotaionHelper(cub)
    IDtrain, IDtest = cub.train_test_id()

    dh = cnn_utils.DeepHelper()

    part_name = 'head'

    pos_x = []
    neg_x = []

    start = dt.now()
    for i, img_id in enumerate(IDtrain):
        if i % 100 == 0:
            print 'done', i, 'in', (dt.now() - start)
        img = caffe.io.load_image(imgs_addr[img_id])
        part_box = imgs_anno.annotation(img_id, part_name)
        if not part_box.is_valid():
            continue
        pos, neg = dh.image_point_features(img, part_box, part_name)

        pos_x.append(pos)
        neg_x.append(neg)

    print 'finished collecting in', (dt.now() - start)
    pos_x = np.vstack(pos_x)
    neg_x = np.vstack(neg_x)

    pos_y = np.ones(pos_x.shape[0], dtype=np.int)
    neg_y = np.zeros(neg_x.shape[0], dtype=np.int)

    X = np.vstack((pos_x, neg_x))
    y = np.concatenate((pos_y, neg_y))

    print 'started learning'
    l_start = dt.now()
    model = sklearn.ensemble.RandomForestClassifier(n_estimators=10, max_depth=20, n_jobs=1, random_state=912)
    model.fit(X, y)

    preds = model.predict_proba(X)
    print sklearn.metrics.auc(y, preds[:, 0])

    joblib.dump(model, 'detectors/%s_model.mdl' % part_name, compress=3)
    print 'learned in', (dt.now() - l_start)
    print 'finsihed in', (dt.now() - start)


if __name__ == '__main__':
    main()
