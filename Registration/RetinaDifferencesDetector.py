import matplotlib.pyplot as plt
import numpy as np
from skimage import segmentation, measure, morphology, filters


PIXELS = 255


class RetinaDifferencesDetctor:
    def __init__(self, registrated_bl_image, registrated_fu_image):
        self.bl_image = registrated_bl_image / PIXELS if np.any(registrated_bl_image > 1) else registrated_bl_image
        self.fu_image = registrated_fu_image / PIXELS if np.any(registrated_fu_image > 1) else registrated_fu_image

    def detect_differences(self):
        """
        This function detects differences of two given retina scans.
        """
        diff = self.fu_image - self.bl_image
        diff[diff > 0] = 0
        diff = np.abs(diff)

        th1 = filters.threshold_yen(diff)
        th2 = filters.threshold_li(diff)
        diff = diff > (th1 + th2) / 2

        diff_label = measure.label(diff)
        diff_label = morphology.binary_dilation(diff_label).astype(int)
        diff_label = segmentation.clear_border(diff_label)
        diff_label = morphology.remove_small_objects(diff_label, 50)
        diff_label = morphology.binary_erosion(diff_label, morphology.disk(3))
        diff_label = filters.median(diff_label)
        diff_label = morphology.remove_small_objects(diff_label.astype(bool), min_size=60, connectivity=8).astype(int)
        diff_label = filters.median(diff_label)
        seg = morphology.binary_closing(diff_label, morphology.disk(20)).astype(int)

        contour = segmentation.find_boundaries(seg)

        fig, axs = plt.subplots(1, 2, figsize=(14, 8))
        for i, img in enumerate(zip((self.bl_image, self.fu_image), ("Baseline image", "Follow up image"))):
            axs[i].imshow(img[0], cmap="gray")
            axs[i].imshow(contour, cmap="Reds", alpha=0.5)
            axs[i].title.set_text(f"{img[1]} with difference contour")
        fig.show()



