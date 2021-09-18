import os

from config import Config


def load(config):
    dirs = os.listdir(config.lfw_root)
    dirs = [os.path.join(config.lfw_root, dir) for dir in dirs]
    dirs = [dir for dir in dirs if os.path.isdir(dir)]
    dir_files = {}
    for dir in dirs:
        dir_files[dir] = len(os.listdir(dir))

    sored_dir_files = [[k, v] for k, v in sorted(dir_files.items(), key=lambda item: item[1])]
    return sored_dir_files




# python plot.py
if __name__ == '__main__':
    config = Config()
    load(config)
