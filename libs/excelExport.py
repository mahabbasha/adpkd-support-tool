import xlsxwriter
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

labels1 = ['MarkierungsNr.', 'Dateiname', 'area', 'perim', 'radius', 'Volumen']
labels2 = ['Dateiname', 'Zellenanzahl'] 


class cellTableGenerator:
    def __init__(self, filename):
        self.wb = xlsxwriter.Workbook('{0}'.format(filename))
        self.ws1 = self.wb.add_worksheet()
        self.ws2 = self.wb.add_worksheet()
        bold = self.wb.add_format({'bold': True})
        for i, l in enumerate(labels1):
            self.ws1.write(4, i, l, bold)
        for i, l in enumerate(labels2):
            self.ws2.write(4, i, l, bold)
        self.ws1.set_column('A:A', 13)
        self.ws1.set_column('B:B', 30)
        self.ws1.set_column('F:F', 15)
        self.ws2.set_column('A:A', 30)
        self.ws2.set_column('B:B', 13)
        self.writer1_row = 6
        self.writer2_row = 6

    def close(self):
        self.wb.close()

    def add_cell(self, idx, image_filename, area, perim, radius, v):
        for c, e in enumerate([idx, image_filename, area, perim, radius, v]):
            self.ws1.write(self.writer1_row, c, e)
        self.writer1_row += 1

    def add_cellcount(self, filename, number):
        for c, e in enumerate([filename, number]):
            self.ws2.write(self.writer2_row, c, e)
        self.writer2_row += 1


class scaleDialog(QDialog):
    def __init__(self, scaling, parent=None, width=0, height=0):
        super(scaleDialog, self).__init__(parent)
        self.width, self.height = 100, 100
        self.pixel_scale = scaling
        self.filename = 'Zystenauswertung'
        layout = QFormLayout()
        self.img_label = QLabel('Bilddimensionen:')
        self.img_dims = QLabel('\t {0}x{1} \t'.format(width, height))
        layout.addRow(self.img_label, self.img_dims)

        self.label = QLabel("Dateiname: \t {0}.xlsx".format(self.filename))
        self.btn = QPushButton('Ändern')
        self.btn.clicked.connect(self.getFilename)
        layout.addRow(self.label, self.btn)

        self.label1 = QLabel("Pixelgröße in µm: \t {0} \t".format(scaling))
        self.btn1 = QPushButton('Ändern')
        self.btn1.clicked.connect(self.getPixelSize)
        layout.addRow(self.label1, self.btn1)

        self.exitBtn = QPushButton('Fertig')
        self.exitBtn.clicked.connect(self.closeIt)
        layout.addRow(self.exitBtn)

        self.setLayout(layout)
        self.setWindowTitle("Skalierungen")
        self.setWindowModality(Qt.WindowModal)
        self.show()

    def getPixelSize(self):
        text, ok = QInputDialog.getText(self, 'Pixelgröße ändern', 'Neuer Wert:')
        if ok and text:
         self.label1.setText("Pixelgröße in µm: \t {0} \t".format(text))
         if ',' in text:
            self.pixel_scale = float(text.replace(',','.'))
         else:
            self.pixel_scale = float(text)


    def getFilename(self):
        text, ok = QInputDialog.getText(self, 'Dateiname ändern', 'Name:')
        if ok and text:
         self.label.setText("Dateiname: \t {0}.xlsx".format(text))
         self.filename = text

    def closeIt(self):
        self.close()
