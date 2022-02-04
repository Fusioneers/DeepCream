import os

time_format = '%Y-%m-%d %H-%M-%S'

logging_format = '%(asctime)s: %(filename)s: %(lineno)d: %(levelname)s: ' \
                 '%(message)s'

default_appr_dist = 3
default_step_len = 2

rep_path = os.path.normpath(
    os.path.dirname(__file__).removesuffix('DeepCream'))
