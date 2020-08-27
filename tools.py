import json
import os
import pickle


def make_dirs(path):
    os.makedirs(path, exist_ok=True)
    return path


def from_json(path):
    with open(path, 'r') as rd:
        data = json.load(rd)

    return data


def from_pickle(path):
    with open(path, 'rb') as rd:
        return pickle.load(rd)


def to_pickle(data, path):
    with open(path, 'wb') as wrt:
        pickle.dump(data, wrt)


def to_json(data, path, indent=None):
    with open(path, 'w') as wrt:
        if indent:
            json.dump(data, wrt, indent=indent)
        else:
            json.dump(data, wrt)


def to_txt(text, path):
    with open(path, 'w') as wrt:
        wrt.write(text)

