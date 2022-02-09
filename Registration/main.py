from RetinaRegistration import RetinaRegistration, REGISTRATION_BY_FEATURES, REGISTRATION_BY_SEGMENTATION
from RetinaDifferencesDetector import RetinaDifferencesDetctor


if __name__ == '__main__':
    retina_registraion = RetinaRegistration("BL03.bmp", "FU03.bmp", REGISTRATION_BY_SEGMENTATION)
    # retina_registraion = RetinaRegistration("BL03.bmp", "FU03.bmp", REGISTRATION_BY_FEATURES)
    bl_warpped, fu_warpped = retina_registraion.execute_registration(REGISTRATION_BY_SEGMENTATION)
    # retina_diff_detector = RetinaDifferencesDetctor(bl_warpped, fu_warpped)
    # retina_diff_detector.detect_differences()
