import sys
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler
import json
from easydict import EasyDict
import os

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)


def setup_logging(log_dir, mode):
    log_file_format = "[%(levelname)s] - %(asctime)s: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    exp_file_handler = RotatingFileHandler(os.path.join(log_dir, '{}.log'.format(mode)), maxBytes=10 ** 6,
                                           backupCount=5)
    stdout_handler = logging.StreamHandler(sys.stdout)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(stdout_handler)