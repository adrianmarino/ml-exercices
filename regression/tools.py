def table_decorator(header, content):
    str = '-' * 80 + '\n'
    str += "{} ({} rows)\n".format(header, len(content))
    str += '-' * 80 + '\n'
    str += content.to_string() + '\n'
    str += '-' * 80
    return str