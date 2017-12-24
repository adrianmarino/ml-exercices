from tabulate import tabulate


class ConfidenceTable:
    def __init__(self, rows=[], format='grid'):
        self.format = format
        self.headers = ['Algorithm', 'Confidence']
        self.rows = self.__to_rows(rows)

    def add_rows(self, rows):
        self.rows.extend(self.__to_rows(rows))
        return self

    def add_row(self, row):
        self.rows.append([row[0], "{}%".format(row[1])])
        return self

    def __to_rows(self, rows):
        return map(lambda row: [row[0], "{}%".format(row[1])],  rows)

    def __str__(self):
        return tabulate(self.rows, headers=self.headers, tablefmt=self.format)
