import os
from easydict import EasyDict

config = EasyDict()
config.WIDER_ROOT = '/home/vijay/datasets/WIDERFACE'
config.WIDER_devkit = os.path.join(config.WIDER_ROOT, 'wider_face_split')
config.WIDER_val = os.path.join(config.WIDER_ROOT, 'WIDER_val')
config.WIDER_train = os.path.join(config.WIDER_ROOT, 'WIDER_train')
config.WIDER_test = os.path.join(config.WIDER_ROOT, 'WIDER_test')
config.WIDER_facepoints = [os.path.join(config.WIDER_ROOT, 'WIDER_annotations', x, 'label.txt') for x in ['train', 'test', 'val']]
