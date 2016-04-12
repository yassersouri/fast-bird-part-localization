import sklearn.externals
import cnn_utils
import geometry_utils
import matplotlib.pylab as plt
import vis_utils


class Detector(object):
    def __init__(self, model_path):
        self.model = sklearn.externals.joblib.load(model_path)
        self.dh = cnn_utils.DeepHelper()

    def detect(self, img):
        """
        A little bit inefficient if one just needs the rectangle and not the probability image.
        """
        self.dh.init_with_image(img)
        Xtest = self.dh.ffeats.reshape(self.dh.ffeats.shape[0] * self.dh.ffeats.shape[1], self.dh.ffeats.shape[2])
        preds = self.model.predict(Xtest)
        preds_prob = self.model.predict_proba(Xtest)
        preds_img = preds.reshape(self.dh.ffeats.shape[0], self.dh.ffeats.shape[1])
        preds_prob_image = preds_prob[:, 1].reshape(self.dh.ffeats.shape[0], self.dh.ffeats.shape[1])
        pred_box = geometry_utils.find_rect_from_preds(geometry_utils.post_process_preds(preds_img))

        return pred_box, preds_prob_image

    def draw(self, img, part_detected, part_probability):
        fig = plt.figure(figsize=(10, 20))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.imshow(part_detected.draw_box(img, color=(1, 0, 0)))
        vis_utils.vis(part_probability, img, ax=ax2, fig=fig)

        plt.show()
