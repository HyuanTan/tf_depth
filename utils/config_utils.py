import numpy as np
import json
from easydict import EasyDict as edict

import log_helper

logger = log_helper.get_logger()

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if b[k] is not None and old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception, e:
                logger.error('Error under config key: {}'.format(k))
                raise e
        else:
            b[k] = v


def cfg_from_file(filename, cfg):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, cfg)


def print_config(cfg):
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('config settings:')
    print(json.dumps(cfg, sort_keys=True, indent=2))
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
