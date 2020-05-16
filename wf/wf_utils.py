import os
import sys
from scipy.io import loadmat
from PIL import Image
import random
import cv2
from config import config
from viz import viz_display

# configurable
WIDER_ROOT = config.WIDER_ROOT
WIDER_devkit = config.WIDER_devkit
WIDER_val = config.WIDER_val
WIDER_train = config.WIDER_train
WIDER_test = config.WIDER_test

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def get_list(_type, full_path=True):
    _file = "wider_face_%s.mat"%_type
    if _type == 'val':
        _activeWIDER = os.path.join(WIDER_val, 'images')
    else:
        _activeWIDER = os.path.join(WIDER_train, 'images')

    _fileMat = os.path.join(WIDER_devkit, _file)
    mat = loadmat(_fileMat)
    
    if full_path:
        return ["%s/%s/%s.jpg\n"%(_activeWIDER, mat['event_list'][i][0][0], x[0][0]) for i in range(len(mat['file_list'])) for j in mat['file_list'][i] for x in j]
    else:
        return ["%s.jpg\n"%(x[0][0]) for i in range(len(mat['file_list'])) for j in mat['file_list'][i] for x in j]
    '''
    _out = []
    for i in range(len(mat['file_list'])):
        for j in mat['file_list'][i]:
            for x in j:
                _out.append("%s/%s/%s.jpg\n"%(_activeWIDER, mat['event_list'][i][0][0], x[0][0]))
    return _out
    '''

def get_list_file(fileout, _type):
    _file = 'wider_face_%s.mat'%_type 
    if _type == 'val':
        _activeWIDER = os.path.join(WIDER_val, 'images')
    elif _type == 'test':
        _activeWIDER = os.path.join(WIDER_test, 'images')
    else:
        _activeWIDER = os.path.join(WIDER_train, 'images')

    _fileMat = os.path.join(WIDER_devkit, _file)
    mat = loadmat(_fileMat)
    
    with open(fileout, 'w') as f:
        for i in range(len(mat['file_list'])):
            for j in mat['file_list'][i]:
                for x in j:
                    f.write("%s/%s/%s.jpg\n"%(_activeWIDER, mat['event_list'][i][0][0] ,x[0][0]))

def get_list_filter(_type, wf_class):
    _file = 'wider_face_%s.mat'%_type 
    if _type == 'val':
        _activeWIDER = os.path.join(WIDER_val, 'images')
    elif _type == 'test':
        _activeWIDER = os.path.join(WIDER_test, 'images')
    else:
        _activeWIDER = os.path.join(WIDER_train, 'images')

    _fileMat = os.path.join(WIDER_devkit, _file)
    mat = loadmat(_fileMat)
    
    try:
        _classIndex = mat['file_list'].index(wf_class)
    except Exception as e:
        print('No class named %s found within %s'%(_file, ))
        return None

    return ["%s/%s/%s.jpg\n"%(_activeWIDER, mat['event_list'][i][0][0], x[0][0]) for i in range(len(mat['file_list'])) for j in mat['file_list'][_classIndex] for x in j]

def get_gt(_type, use_mat=False):
    _file = 'wider_face_%s.mat'%_type 
    if _type == 'val':
        _activeWIDER = os.path.join(WIDER_val, 'images')
    elif _type == 'test':
        _activeWIDER = os.path.join(WIDER_test, 'images')
    else:
        _activeWIDER = os.path.join(WIDER_train, 'images')

    if use_mat:
        _fileMat = os.path.join(WIDER_devkit, _file)
        mat = loadmat(_fileMat)
        # TODO: Finish mat parsing for bbx gt later:

    else:
        gt_file = os.path.join(WIDER_devkit, 'wider_face_%s_bbx_gt.txt'%_type)
        gt_struct = dict() #keyed on filename (without extension)
        with open(gt_file, 'r') as f:
            wf_raw = f.readlines()
            # Replace later with list comprehension?
            index = 0
            for i, line in enumerate(wf_raw):
                line = line.strip()
                if index == 0:
                    # add to dict:
                    entry_line = line
                    gt_struct[entry_line] = []
                    index = 1
                elif index == 1:
                    _timetoloop = int(line.strip())+i
                    index = 2
                elif index == 2:
                    if _timetoloop == i:
                        index = 0

                    _setup = [int(_i) for _i in line.split(' ')]
                    _entry = {"x1": _setup[0], "y1": _setup[1], "w": _setup[2], "h": _setup[3], "blur": _setup[4], "expression": _setup[5], "illumination": _setup[6], "invalid": _setup[7]}
                    gt_struct[entry_line].append(_entry)
        return gt_struct

# TODO: Make yolo converter for WIDERFACE:
def wf_yolo_convert(_type):
    if _type == 'val':
        _activeWIDER = os.path.join(WIDER_val, 'images')
    elif _type == 'train':
        _activeWIDER = os.path.join(WIDER_train, 'images')

    gt_wf = get_gt(_type)
    gt_yolo = dict()

    for item in gt_wf.items():
        _img = Image.open(os.path.join(_activeWIDER, item[0]))
        img_w, img_h = _img.size
        gt_yolo[item[0]] = []
        for obj in item[1]:
            # untested:
            x1,y1,x2,y2 = obj['x1'], obj['y1'], obj['x1']+obj['w'], obj['y1']+obj['h']
            gt_yolo[item[0]].append(convert((img_w, img_h), (x1,x2,y1,y2)))

    return gt_yolo

def grab_random(_type='val', seed=10):
    _val = get_list(_type)
    random.seed(seed)
    return random.choice(_val)

# debugging only:
if __name__ == '__main__':
    #print(get_list('val'))
    #get_list_file('wf_train.txt','train')
    #get_gt('val')
    #wf_yolo_convert('val')
    randoImg = grab_random(seed=30)
    gt_list = get_gt('val')
    viz_display(os.path.join(WIDER_val,'images',randoImg), gt_list)
