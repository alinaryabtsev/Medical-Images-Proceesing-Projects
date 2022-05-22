from RetinaRegistration import RetinaRegistration, REGISTRATION_BY_FEATURES, REGISTRATION_BY_SEGMENTATION
from RetinaDifferencesDetector import RetinaDifferencesDetctor
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import os
import time

BASELINE = "BL"
FOLLOW_UP = "FU"
IMAGES_DIR = "RetinaImages"


def running_registration_on_files(baseline_images, follow_up_images, registration_type):
    """
    This function runs in paralel registration on given baseline and follow up images.
    :param baseline_images: a list of baseline images
    :param follow_up_images: a list of follow up images
    :param registration_type: 1 if registration by features, 2 if registration by segmentation
    :return: a list of couples of baseline and follow up warpped images
    """
    t1 = time.time()
    with ProcessPoolExecutor() as executor:
        registrators = executor.map(RetinaRegistration, baseline_images, follow_up_images,
                                    repeat(registration_type))
    with ProcessPoolExecutor() as executor:
        bl_fu_warpeed = executor.map(RetinaRegistration.execute_registration, registrators,
                                     repeat(registration_type))
    print(f"Took {time.time() - t1} secs")
    return bl_fu_warpeed


def running_diffrences_detector_on_files(bl_warpped_images, fu_warpped_images):
    """
    This function runs in paralel diffrences detctor on given baseline and follow up images.
    :param bl_warpped_images: a list of baseline warpped images
    :param fu_warpped_images: a list of follow up warpped images
    """
    t1 = time.time()
    # retina_diff_detector = RetinaDifferencesDetctor(bl_warpped, fu_warpped)
    # retina_diff_detector.detect_differences()
    with ProcessPoolExecutor() as executor:
        detectors = executor.map(RetinaDifferencesDetctor, bl_warpped_images, fu_warpped_images)
    with ProcessPoolExecutor() as executor:
        bl_fu_warpeed = executor.map(RetinaDifferencesDetctor.detect_differences, detectors)
    print(f"Took {time.time() - t1} secs")
    return bl_fu_warpeed


if __name__ == '__main__':
    bl_images = sorted([os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) if f.startswith(BASELINE)])
    fu_images = sorted([os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) if f.startswith(FOLLOW_UP)])
    bl_fu_warpeed_ccouples = list(running_registration_on_files(bl_images, fu_images, REGISTRATION_BY_FEATURES))
    running_diffrences_detector_on_files((c[0] for c in bl_fu_warpeed_ccouples), (c[1] for c in bl_fu_warpeed_ccouples))
