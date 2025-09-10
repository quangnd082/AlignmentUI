import sys
sys.path.append('Libs')
from cameras.MVSImport.LoadAndSave import save_feature
from cameras.hik import HIK
from cameras.soda import SODA
from cameras.webcam import Webcam

def get_camera_devices():
    return HIK.get_devices()


# if __name__ == '__main__':
#     save_feature()