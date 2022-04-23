from client_module.log import logger as l
from client_module.log import init_logging
import scripts.helpful_scripts as script
import json
import argparse
import os,sys


class Node(object):

    def __init__(self, setting_path):
        setting = script.read_yaml(setting_path)
        init_logging(log_level_str=setting['log_level'])

    def run(self):
        pass


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Node"
)
parser.add_argument('-s', '--setting_path', type=str, help='the setting file\'s path')
if __name__ == "__main__":
    args = parser.parse_args()
    setting_path = args.setting_path
    print('create node with setting_path:', setting_path)
