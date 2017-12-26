def singleton(cls):
    instances = {}

    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return get_instance


def table_decorator(header, content):
    str = '-' * 80 + '\n'
    str += "{} ({} rows)\n".format(header, len(content))
    str += '-' * 80 + '\n'
    str += content.to_string() + '\n'
    str += '-' * 80
    return str