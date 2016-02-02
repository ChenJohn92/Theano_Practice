import os

UTIL_FOLDER = os.path.abspath(".")


def GetBaseDir():
    return os.path.join(UTIL_FOLDER, "..")

def GetUtilDir():
    return UTIL_FOLDER

def GetDataDir():
    return os.path.join(UTIL_FOLDER, "../Datasets/")