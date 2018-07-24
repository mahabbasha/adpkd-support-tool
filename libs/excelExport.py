import xlsxwriter
labels = ['MarkierungsNr.', 'Dateiname', 'area', 'perim', 'radius', 'Volumen']


class cellTableGenerator:
    def __init__(self, filename):
        self.wb = xlsxwriter.Workbook('{0}'.format(filename))
        self.ws = self.wb.add_worksheet()
        bold = self.wb.add_format({'bold': True})
        for i, l in enumerate(labels):
            self.ws.write(4, i, l, bold)
        self.ws.set_column('A:A', 13)
        self.ws.set_column('B:B', 30)
        self.ws.set_column('F:F', 15)
        self.writer_row = 6

    def close(self):
        self.wb.close()

    def add_cell(self, idx, image_filename, area, perim, radius, v):
        for c, e in enumerate([idx, image_filename, area, perim, radius, v]):
            self.ws.write(self.writer_row, c, e)
        self.writer_row += 1
