import os
import wf_utils
from config import config
from scipy.io import loadmat

WF_EVAL_PATH = config.WIDER_eval_gt

def get_eval_files(_type='easy'):
    _mat = loadmat(os.path.join(WF_EVAL_PATH, "wider_%s_val.mat"%_type))

