import os
import sys
from config import config

def load_label():
    # replace later:
    _index = config.WIDER_facepoints[0]
    # test and val have no cordinates in it so force train only
    with open(_index, 'r') as f:
        return f.readlines()

def get_landmarks():
    raw_out = load_label()
    flip = 0
    _out = dict()

    for i in raw_out:
        if i[0:2] == '# ':
            _entry = i.split('# ')[-1].strip()
            _out[_entry] = []
        else:
            data = [float(x) for x in i.split(' ')]
            # rect coordinate = r
            # face coordinate = c
            # r1 r2 r3 r4 c1 c2 conf. c1 c2 conf. c1 c2 conf. c1 c2 conf. c1 c2 conf.
            c1, c2, c3, c4, c5, conf = data[4:6], data[7:9], data[10:12], data[13:15], data[16:18], data[-1]
            packaged = {'bbox': data[0:4], 'landmarks': (c1,c2,c3,c4,c5), 'conf': conf}
            _out[_entry].append(packaged)
    return _out

if __name__ == '__main__':
    _out = get_landmarks()
    from viz import viz_landmarks
    from wf_utils import grab_random
    viz_landmarks(os.path.join(config.WIDER_train,'images',grab_random(_type='train')), _out)
