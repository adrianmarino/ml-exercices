import pickle
import os


def singleton(cls):
    instances = {}

    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return get_instance


def first(collection): return list(collection)[0]


def table_decorator(header, content):
    str = '-' * 80 + '\n'
    str += "{} ({} rows)\n".format(header, len(content))
    str += '-' * 80 + '\n'
    str += content.to_string() + '\n'
    str += '-' * 80
    return str


class ObjectStorage:
    def __init__(self, base_path="./storage"):
        self.base_path = base_path

    def save(self, object, filename=''):
        if not filename:
            filename = object.name()

        os.makedirs(self.base_path, exist_ok=True)
        with open(self.__path(filename), 'wb+') as out_stream:
            pickle.dump(object, out_stream)

    def load(self, _class, filename=''):
        if not filename:
            filename = _class.__name__

        in_stream = open(self.__path(filename), 'rb')
        return pickle.load(in_stream)

    def __path(self, filename): return '{}/{}.pickle'.format(self.base_path, filename)