#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import codecs
# import heapq
import os
import os.path
import platform
# import re
# import subprocess
import sys
import glob
# import xml.etree.ElementTree as ET
# from collections import defaultdict
from functools import partial
from PIL import Image, ImageFont, ImageDraw

import cv2
import numpy as np

# import skimage
from skimage import io
from skimage import img_as_float
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
from skimage.filters.rank import enhance_contrast
from skimage.morphology import disk
from skimage._shared.utils import assert_nD
# from skimage.draw import line, line_aa
from shapely.geometry import Polygon

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

# import resources

from libs.canvas import Canvas
from libs.colorDialog import ColorDialog
from libs.constants import *
from libs.labelDialog import LabelDialog
from libs.labelFile import LabelFile
from libs.labelFile import LabelFileError
from libs.lib import addActions
# from libs.lib import fmtShortcut
from libs.lib import generateColorByText
from libs.lib import newAction
# from libs.lib import newIcon
from libs.lib import struct
from libs.pascal_voc_io import PascalVocReader
from libs.pascal_voc_io import PascalVocWriter
from libs.pascal_voc_io import XML_EXT
from libs.settings import Settings
from libs.shape import DEFAULT_FILL_COLOR
from libs.shape import DEFAULT_LINE_COLOR
from libs.shape import Shape
from libs.toolBar import ToolBar
from libs.ustr import ustr
# from libs.version import __version__
from libs.zoomWidget import ZoomWidget

from libs.detection import YOLODetector, OBJ_THRESH
from libs.detection import MaskRCNNDetector
from libs.detection import UNetSegmentation
from libs.excelExport import cellTableGenerator, scaleDialog

__appname__ = 'ADPKD Support Tool'

# Utility functions and classes.


def have_qstring():
    '''p3/qt5 get rid of QString wrapper as py3 has native unicode str type'''
    return not (sys.version_info.major >= 3 or QT_VERSION_STR.startswith('5.'))


def util_qt_strlistclass():
    return QStringList if have_qstring() else list


class WindowMixin(object):

    def menu(self, title, actions=None):
        if platform.uname().system.startswith('Darw'):
            self._menu_bar = QMenuBar()
        else:
            self._menu_bar = self.menuBar()
        # menu = self.menuBar().addMenu(title)
        menu = self._menu_bar.addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


# PyQt5: TypeError: unhashable type: 'QListWidgetItem'
class HashableQListWidgetItem(QListWidgetItem):

    def __init__(self, *args):
        super(HashableQListWidgetItem, self).__init__(*args)

    def __hash__(self):
        return hash(id(self))


class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, defaultFilename=None, defaultPrefdefClassFile=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # Load setting in the main thread
        self.settings = Settings()
        self.settings.load()
        settings = self.settings
        # self.mask_model_weights = 'yolo3_cells_2.h5'
        self.mask_model_weights = 'mask_rcnn_cell_0030_4.h5'
        self.unet_model_weights = 'unet_cells.hdf5'
        # Save as Pascal voc xml
        self.defaultSaveDir = None
        self.usingPascalVocFormat = True
        # For loading all image under a directory
        self.mImgList = []
        self.dirname = None
        self.labelHist = []
        self.lastOpenDir = None

        # Whether we need to save or not.
        self.dirty = False
        self._noSelectionSlot = False
        self.autoSaving = True
        self.singleClassMode = True
        self.lastLabel = None

        # Load predefined classes to the list
        self.loadPredefinedClasses(defaultPrefdefClassFile)

        # Main widgets and related state.
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)

        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.prevLabelText = ''

        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)

        # Create a widget for using default label
        self.useDefaultLabelCheckbox = QCheckBox(u'Use default label')
        self.useDefaultLabelCheckbox.setChecked(False)
        self.defaultLabelTextLine = QLineEdit()
        useDefaultLabelQHBoxLayout = QHBoxLayout()
        useDefaultLabelQHBoxLayout.addWidget(self.useDefaultLabelCheckbox)
        useDefaultLabelQHBoxLayout.addWidget(self.defaultLabelTextLine)
        useDefaultLabelContainer = QWidget()
        useDefaultLabelContainer.setLayout(useDefaultLabelQHBoxLayout)

        # Create a widget for edit and diffc button
        self.diffcButton = QCheckBox(u'difficult')
        self.diffcButton.setChecked(False)
        self.diffcButton.stateChanged.connect(self.btnstate)
        self.editButton = QToolButton()
        self.editButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Add some of widgets to listLayout
        listLayout.addWidget(self.editButton)
        listLayout.addWidget(self.diffcButton)
        listLayout.addWidget(useDefaultLabelContainer)

        # Tzutalin 20160906 : Add file list and dock to move faster
        self.fileListWidget = QListWidget()
        self.fileListWidget.itemDoubleClicked.connect(self.fileitemDoubleClicked)
        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        filelistLayout.addWidget(self.fileListWidget)
        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.filedock = QDockWidget(u'File List', self)
        self.filedock.setObjectName(u'Files')
        self.filedock.setWidget(fileListContainer)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)

        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.saveFileSignal.connect(self.saveFile)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scroll)
        # self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        # Tzutalin 20160906 : Add file list and dock to move faster
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedock)
        self.filedock.setFeatures(QDockWidget.DockWidgetFloatable)

        self.dockFeatures = QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
        # self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)
        self.showSegmentationOverlay = None
        self.segmentationOverlay = None
        # Custom Cell Detector

        # self.detector = YOLODetector(self.mask_model_weights)
        self.detector = MaskRCNNDetector(self.mask_model_weights)
        self.unet_seg = UNetSegmentation(self.unet_model_weights)
        # Actions
        action = partial(newAction, self)
        # quit = action('&Schließen', self.close, 'Ctrl+Q', 'Schließen', u'Anwendung verlassen')
        opendir = action('&Ordner\nöffnen', self.openDirDialog, 'Ctrl+u', 'icons/open.png', u'Ordner öffnen')
        openNextImg = action('&Nächstes\nBild', self.openNextImg, 'd', 'icons/next.png', u'Nächstes Bild anzeigen')
        openPrevImg = action('&Voheriges\nBild', self.openPrevImg, 'a', 'icons/prev.png', u'Voheriges Bild anzeigen')
        save = action('&Speichern', self.saveFile, 'Ctrl+S', 'icons/save.png', u'Speichern der Markierungen', enabled=False)
        close = action('&Schließen', self.closeFile, 'Ctrl+W', 'icons/close.png', u'Aktuelle Datei schließen')
        resetSettings = action('&Zurücksetzen\naller\nEinstellungen &\n Neu starten', self.resetAll, None, 'icons/resetall.png', u'Einstellungen zurücksetzen')
        createMode = action('&Markierung\nerstellen', self.setCreateMode, 'w', 'icons/feBlend-icon.png', u'Markierungsmodus', enabled=False)
        editMode = action('&Markierungen\nbearbeiten', self.setEditMode, 'Ctrl+J', 'icons/edit.png', u'Editierungsmodus', enabled=False)
        delete = action('&Markierungen\nlöschen', self.deleteSelectedShape, 'delete', 'icons/delete.png', u'Löschen', enabled=False)
        reload = action('&Bild neu laden', self.reloadImg, 'Ctrl+R', 'icons/verify.png', u'Aktuelle Bild neu laden', enabled=True)
        resetBoxes = action('&Markierungen\nzurücksetzen', self.resetImg, None, 'icons/quit.png', u'Markierungen des aktuellen Bildes zurücksetzen', enabled=True)
        segmentationOverlay = action('Segmentierung einblenden', self.toggleSegmentationOverlay, None, 'Overlay einblenden', u'Segmentierung einblenden', checkable=True, enabled=False)
        contourOverlay = action('Konturmodus', self.toggleContourOverlay, 'Ctrl+Shift+C', 'Overlay einblenden', u'Kontur einblenden', checkable=True, enabled=True)
        unet_usage = action('UNet verwenden', self.toggleUnet, None, 'UNet zum Segmentieren verwenden', u'UNet verwenden', checkable=True, enabled=True, checked=True)
        generateOutput = action('Ergebnis erzeugen', self.genOutput, None, 'icons/labels.png', u'Ergebnisbild erzeugen')
        autoDetect = action('&Automatische\nErkennung', self.cellDetection, None, 'icons/zoom.png', u'Automatische Erkennung von Zellen')
        autoDetectDir = action('&Automatische\nErkennung des Ordners', self.cellDetectionDir, None, 'icons/zoom.png', u'Automatische Erkennung von Zellen des gesamten Ordners')
        zoomIn = action('Zoom &In', partial(self.addZoom, 10), 'Ctrl++', 'zoom-in', u'Increase zoom level', enabled=False)
        zoomOut = action('&Zoom Out', partial(self.addZoom, -10), 'Ctrl+-', 'zoom-out', u'Decrease zoom level', enabled=False)
        zoomOrg = action('&Original size', partial(self.setZoom, 100), 'Ctrl+=', 'zoom', u'Zoom to original size', enabled=False)
        fitWindow = action('&Fit Window', self.setFitWindow, 'Ctrl+F', 'fit-window', u'Zoom follows window size', checkable=True, enabled=False)
        fitWidth = action('Fit &Width', self.setFitWidth, 'Ctrl+Shift+F', 'fit-width', u'Zoom follows window width', checkable=True, enabled=False)
        # # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut,
                       zoomOrg, fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        # Store actions for further handling
        self.actions = struct(save=save,
        close=close, resetSettings=resetSettings,
        delete=delete,
        createMode=createMode, editMode=editMode,
        autoDetect=autoDetect,
        autoDetectDir=autoDetectDir,
        zoomActions=zoomActions,
        generateOutput=generateOutput,
        segmentationOverlay=segmentationOverlay,
        contourOverlay=contourOverlay,
        advancedContext=(delete, contourOverlay),
        onLoadActive=(close, createMode, editMode))

        self.menus = struct(
            overlays=self.menu('&Konturen'))

        addActions(self.menus.overlays, (segmentationOverlay, contourOverlay, unet_usage))
        addActions(self.canvas.menus[0], self.actions.advancedContext)
        addActions(self.canvas.menus[1], [action('&Move here', self.moveShape)])
        self.tools = self.toolbar('Tools')

        self.actions.advanced = (opendir, openNextImg, openPrevImg, createMode, autoDetectDir, autoDetect, save, reload, None, editMode, delete, generateOutput, None,  resetBoxes, resetSettings)

        # Application state.
        self.image = QImage()
        self.filePath = ustr(defaultFilename)
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False
        self.difficult = False

        ## Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [ustr(i) for i in recentFileQStringList]
            else:
                self.recentFiles = recentFileQStringList = settings.get(SETTING_RECENT_FILES)

        self.pixel_scale = settings.get(SETTING_PIXEL_SCALING, 0)
        self.unet_usage = settings.get(SETTING_UNET_USAGE, True)
        size = settings.get(SETTING_WIN_SIZE, QSize(600, 500))
        position = settings.get(SETTING_WIN_POSE, QPoint(0, 0))
        self.resize(size)
        self.move(position)
        ####### Automatic Save Context
        saveDir = ustr(settings.get(SETTING_SAVE_DIR, None))
        self.lastOpenDir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))
        if saveDir is not None and os.path.exists(saveDir):
            self.defaultSaveDir = saveDir
            self.statusBar().showMessage('%s gestartet. Annotation werden in %s gespeichert' % (__appname__, saveDir))
            self.statusBar().show()

        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        Shape.line_color = self.lineColor = DEFAULT_LINE_COLOR
        Shape.fill_color = self.fillColor = DEFAULT_FILL_COLOR
        self.canvas.setDrawingColor(self.lineColor)
        # Add chris
        Shape.difficult = self.difficult

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(settings.get(SETTING_ADVANCE_MODE, False)):
            pass

        # Since loading the file may take some time, make sure it runs in the background.
        if self.filePath and os.path.isdir(self.filePath):
            self.queueEvent(partial(self.importDirImages, self.filePath or ""))
        elif self.filePath:
            self.queueEvent(partial(self.loadFile, self.filePath or ""))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # Display cursor coordinates at the right of status bar
        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        # Open Dir if deafult file
        if self.filePath and os.path.isdir(self.filePath):
            pass
            # self.openDirDialog(dirpath=self.filePath)

########################################################################################################
########################################################################################################
    ## Support Functions ##

    def noShapes(self):
        return not self.itemsToShapes

    def populateModeActions(self):
        tool, menu = self.actions.advanced, self.actions.advancedContext
        self.tools.clear()
        addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], menu)
        # self.menus.edit.clear()
        actions = (self.actions.createMode, self.actions.editMode)
        # addActions(self.menus.edit, actions + self.actions.editMenu)

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        #self.actions.create.setEnabled(True)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.toggleSegmentationOverlay(False, True)
        self.actions.contourOverlay.setChecked(self.canvas.showContourOverlay)
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        #self.labelList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()

    def currentItem(self):
        #items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    ## Callbacks ##
    # def showTutorialDialog(self):
    #     subprocess.Popen([self.screencastViewer, self.screencast])

    def createShape(self):
        # assert self.beginner()
        self.canvas.setEditing(False)
        # self.actions.create.setEnabled(False)

    def resetOverlays(self):
        self.showSegmentationOverlay = False
        self.segmentationOverlay = None
        self.actions.segmentationOverlay.setChecked(False)
        self.actions.contourOverlay.setChecked(self.canvas.showContourOverlay)

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)
        if not drawing: # and self.beginner():
            # Cancel creation.
            self.statusBar().showMessage('Zeichnen abgebrochen')
            self.statusBar().show()
            self.canvas.setEditing(True)
            self.actions.editMode.setEnabled(False)
            self.actions.delete.setEnabled(False)
            self.actions.createMode.setEnabled(True)
            self.canvas.restoreCursor()
            # self.actions.create.setEnabled(True)

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        # self.actions.editMode.setEnabled(not edit)

    def setCreateMode(self):
        self.toggleDrawMode(False)
        self.actions.delete.setEnabled(False)
        self.actions.editMode.setEnabled(True)

    def setEditMode(self):
        self.toggleDrawMode(True)
        self.actions.delete.setEnabled(True)
        self.actions.editMode.setEnabled(False)
        # self.labelSelectionChanged()

    def toggleSegmentationOverlay(self, show=False, reset=False):
        # print('Toggle segmentation called: {}'.format(show))
        if reset:
            # print('but in Reset Mode, ignoring')
            return
        else:
            self.showSegmentationOverlay = show
            # self.renderOverlays()
            self.reloadImg()
        # print('segmentation overlay toggled {}'.format(show))

    def toggleContourOverlay(self, show=False):
        # print('Toggle contour called: {}'.format(show))
        # if reset:
        #     # print('but in Reset Mode, ignoring')
        #     return
        # else:

        if show:
            self.calcContours()
        self.canvas.showContourOverlay = show
        self.canvas.update()
            # self.renderOverlays()
            # self.reloadImg()

    def toggleUnet(self, show=True):
        self.unet_usage = show

    def fileitemDoubleClicked(self, item=None):
        currIndex = self.mImgList.index(ustr(item.text()))
        if currIndex < len(self.mImgList):
            filename = self.mImgList[currIndex]
            if filename:
                self.loadFile(filename)

    # Add chris
    def btnstate(self, item=None):
        """ Function to handle difficult examples Update on each object """
        if not self.canvas.editing():
            return
        # item = self.currentItem()
        # if not item: # If not selected Item, take the first one
        #     item = self.labelList.item(self.labelList.count()-1)
        difficult = self.diffcButton.isChecked()
        try:
            shape = self.itemsToShapes[item]
        except:
            pass
        # Checked and Update
        try:
            if difficult != shape.difficult:
                shape.difficult = difficult
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    # React to canvas signals.
    def shapeSelectionChanged(self, selected=False):
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selectedShape
            if shape:
                self.shapesToItems[shape].setSelected(True)
        self.actions.delete.setEnabled(selected)

    def addLabel(self, shape):
        item = HashableQListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setBackground(generateColorByText(shape.label))
        self.itemsToShapes[item] = shape
        self.shapesToItems[shape] = item

    def remLabel(self, shape):
        if shape is None:
            return
        item = self.shapesToItems[shape]
        del self.shapesToItems[shape]
        del self.itemsToShapes[item]

    def loadLabels(self, shapes):
        s = []
        for label, points, line_color, fill_color, difficult, contour_points, confidence, contourEdited in shapes:
            shape = Shape(label=label)
            for x, y in points:
                shape.addPoint(QPointF(x, y))
            if contour_points:
                for x, y in contour_points:
                    shape.addContourPoint((x, y))
            shape.difficult = difficult
            shape.confidence = confidence
            shape.contourEdited = contourEdited
            shape.close()
            s.append(shape)
            if line_color:
                shape.line_color = QColor(*line_color)
            else:
                shape.line_color = generateColorByText(label)
            if fill_color:
                shape.fill_color = QColor(*fill_color)
            else:
                shape.fill_color = generateColorByText(label)
            self.addLabel(shape)
        self.canvas.loadShapes(s)

    def saveLabels(self, annotationFilePath):
        annotationFilePath = ustr(annotationFilePath)
        if self.labelFile is None:
            self.labelFile = LabelFile()
            self.labelFile.verified = self.canvas.verified

        def format_shape(s):
            return dict(label=s.label,
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(p.x(), p.y()) for p in s.points],
                        difficult=s.difficult,
                        contour_points=s.contour_points,
                        confidence=s.confidence,
                        contourEdited=s.contourEdited)

        shapes = [format_shape(shape) for shape in self.canvas.shapes]
        # Can add differrent annotation formats here
        try:
            if self.usingPascalVocFormat is True:
                print('Img: ' + self.filePath + ' -> Its xml: ' + annotationFilePath)
                self.labelFile.savePascalVocFormat(annotationFilePath, shapes, self.filePath, self.imageData, self.lineColor.getRgb(), self.fillColor.getRgb())
            else:
                self.labelFile.save(annotationFilePath, shapes, self.filePath, self.imageData, self.lineColor.getRgb(), self.fillColor.getRgb())
            return True
        except LabelFileError as e:
            self.errorMessage(u'Error saving label data', u'<b>%s</b>' % e)
            return False

    def labelSelectionChanged(self):
        item = self.currentItem()
        if item and self.canvas.editing():
            self._noSelectionSlot = True
            self.canvas.selectShape(self.itemsToShapes[item])
            shape = self.itemsToShapes[item]
            # Add Chris
            self.diffcButton.setChecked(shape.difficult)

    def labelItemChanged(self, item):
        shape = self.itemsToShapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            shape.line_color = generateColorByText(shape.label)
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

##################################################################################################################################
##################################################################################################################################
    # Callback functions:
    def newShape(self):
        """Pop-up and give focus to the label editor.
        position MUST be in global coordinates.
        """
        text = 'cell'
        self.diffcButton.setChecked(False)
        if text is not None:
            generate_color = generateColorByText(text)
            shape = self.canvas.setLastLabel(text, generate_color, generate_color)
            self.addLabel(shape)
            self.actions.editMode.setEnabled(True)
            self.actions.delete.setEnabled(False)
            self.setDirty()
        else:
            self.canvas.resetAllLines()

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def zoomRequest(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)
        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()
        w = self.scrollArea.width()
        h = self.scrollArea.height()
        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)
        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)
        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)
        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max
        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max
        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def loadFile(self, filePath=None, overlays=None):
        """Load the specified file, or the last opened file if None."""
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)
        # Make sure that filePath is a regular python string, rather than QString
        filePath = str(filePath)
        unicodeFilePath = ustr(filePath)
        # Highlight the file item
        if unicodeFilePath and self.fileListWidget.count() > 0:
            index = self.mImgList.index(unicodeFilePath)
            fileWidgetItem = self.fileListWidget.item(index)
            fileWidgetItem.setSelected(True)
        if unicodeFilePath and os.path.exists(unicodeFilePath):
            # Image now loading with OpenCV function, so it can be modified with OpenCV functions before rendering
            img = read(unicodeFilePath, None)
            # reload image with overlays rendered in it
            overlay = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            overlays = [self.segmentationOverlay]
            for o in overlays:
                if o is not None:
                    overlay = cv2.addWeighted(overlay, 1.0, o, 1.0, 1.0)
                else:
                    continue
            img = cv2.addWeighted(img, 1.0, overlay, 1.0, 0.0)
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            image = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.labelFile = None
            if image.isNull():
                self.errorMessage(u'Error opening file', u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self.image = image
            self.filePath = unicodeFilePath
            self.canvas.loadPixmap(QPixmap.fromImage(image))
            if self.labelFile:
                self.loadLabels(self.labelFile.shapes)
            self.setClean()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            # self.addRecentFile(self.filePath)
            self.toggleActions(True)
            # Label xml file and show bound box according to its filename
            if self.usingPascalVocFormat is True:
                if self.defaultSaveDir is not None:
                    basename = os.path.basename(os.path.splitext(self.filePath)[0]) + XML_EXT
                    xmlPath = os.path.join(self.defaultSaveDir, basename)
                    self.loadPascalXMLByFilename(xmlPath)
                else:
                    xmlPath = os.path.splitext(filePath)[0] + XML_EXT
                    if os.path.isfile(xmlPath):
                        self.loadPascalXMLByFilename(xmlPath)
            self.setWindowTitle(__appname__ + ' ' + filePath)
            self.canvas.setFocus(True)
            return True
        return False

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull()\
           and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if self.dirty:
            self.saveFile()
        settings = self.settings
        # If it loads images from dir, don't load it at the begining
        if self.dirname is None:
            settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
        else:
            settings[SETTING_FILENAME] = ''
        settings[SETTING_WIN_SIZE] = self.size()
        settings[SETTING_WIN_POSE] = self.pos()
        settings[SETTING_WIN_STATE] = self.saveState()
        settings[SETTING_LINE_COLOR] = self.lineColor
        settings[SETTING_FILL_COLOR] = self.fillColor
        settings[SETTING_RECENT_FILES] = self.recentFiles
        settings[SETTING_PIXEL_SCALING] = self.pixel_scale
        settings[SETTING_UNET_USAGE] = self.unet_usage
        if self.defaultSaveDir and os.path.exists(self.defaultSaveDir):
            settings[SETTING_SAVE_DIR] = ustr(self.defaultSaveDir)
        else:
            settings[SETTING_SAVE_DIR] = ""
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
        else:
            settings[SETTING_LAST_OPEN_DIR] = ""
        settings[SETTING_AUTO_SAVE] = self.autoSaving
        settings[SETTING_SINGLE_CLASS] = self.singleClassMode
        settings.save()
###############################################################################################################################################
###############################################################################################################################################

    # User Dialogs
    def reloadImg(self):
        if not self.mayContinue():
            return
        self.renderOverlays()
        print('Reload image')
        self.loadFile(self.filePath)

    def resetImg(self):
        if self.filePath is not None:
            anno_file = self.filePath.split('.')[0] + '.xml'
            if not os.path.exists(anno_file):
                self.noAnnotationFileDialog()
                return
        os.remove(anno_file)
        print('Reset image')
        self.reloadImg()

    def cellDetection(self):
        if not self.mayContinue():
            return
        print(self.filePath)
        # if path is None:
        #     currentPath = self.filePath
        # else:
        #     currentPath = path
        currentPath = self.filePath
        localPath = self.filePath.split(os.path.basename(currentPath))[0]
        imgFileName = os.path.basename(currentPath)
        currentImg = io.imread(currentPath)
        if isinstance(self.detector, YOLODetector):
            boxes = self.detector.predictBoxes(currentImg)
            height, width, depth = currentImg.shape
            filename = currentPath.split('.')[0] + '.xml'
            writer = PascalVocWriter('{0}'.format(localPath), imgFileName, [height, width, depth], localImgPath=currentPath)
            writer.verified = False
            for box in boxes:
                if box.classes[0] > OBJ_THRESH:
                    xmin = int(box.xmin)
                    ymin = int(box.ymin)
                    xmax = int(box.xmax)
                    ymax = int(box.ymax)
                    confidence = box.get_score()
                    writer.addBndBox(xmin, ymin, xmax, ymax, 'cell', 0, [], confidence, False)
                else:
                    continue
        elif isinstance(self.detector, MaskRCNNDetector):
            boxes = self.detector.predictBoxesAndContour(currentImg)
            height, width, depth = currentImg.shape
            filename = currentPath.split('.')[0] + '.xml'
            writer = PascalVocWriter('{0}'.format(localPath), imgFileName, [height, width, depth], localImgPath=currentPath)
            writer.verified = False
            for box in boxes:
                xmin = box.xmin
                xmax = box.xmax
                ymin = box.ymin
                ymax = box.ymax
                contour = box.contour
                confidence = box.confidence
                writer.addBndBox(xmin, ymin, xmax, ymax, 'cell', 0, contour, confidence, False)
        writer.save(targetFile=filename)
        self.loadRecent(currentPath, True)

    def cellDetectionDir(self):
        # print(self.mImgList)
        progress = QProgressDialog('Erkenne Zellen {0}/{1}'.format(0, len(self.mImgList)), None, 0, 0, self)
        progress.setWindowTitle('Bitte warten')
        progress.setWindowModality(Qt.WindowModal)
        progress.setRange(0, len(self.mImgList))
        progress.setValue(0)
        progress.forceShow()
        for p in self.mImgList:
            progress.setLabelText('Erkenne Zellen {0}/{1}'.format(progress.value() + 1, len(self.mImgList)))
            progress.forceShow()
            self.filePath = p
            self.cellDetection()
            progress.setValue(progress.value() + 1)
        progress.close()
        progress = QMessageBox.information(self, u'Information', 'Erkennnung der Zellen abgeschlossen')

    def calcContours(self):
        # from OpenCV contour examples
        rendered = False
        if not self.canvas.shapes:
            return
        else:
            # contourOverlay = np.zeros((self.image.height(), self.image.width(), 3), dtype=np.uint8)
            fullImg = cv2.imread(self.filePath, cv2.IMREAD_COLOR)
            for i, s in enumerate(self.canvas.shapes):
                xmin, ymin = int(s.points[0].x()), int(s.points[0].y())
                xmax, ymax = int(s.points[2].x()), int(s.points[2].y())
                img = fullImg[ymin:ymax + 1, xmin:xmax + 1, :]
                # outerContour = np.zeros((img.shape[0], img.shape[1], 3)).astype(np.uint8)

                if self.canvas.shapes[i].contour_points:
                    continue
                else:
                    rendered = True
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    try:
                        assert_nD(img, 2)
                        if img.shape[0] <= 1 or img.shape[1] <= 1:
                            raise ValueError('Box ist vertikale oder horizontale Linie')
                    except ValueError as e:
                        print(e)
                        print(img, xmin, xmax, ymin, ymax)
                        print('Möglicherweise konnten nicht alle Konturen berechnet werden')
                        self.statusBar().showMessage('Möglicherweise konnten nicht alle Konturen berechnet werden (0)')
                        self.statusBar().show()
                        continue
                    if self.unet_usage:
                        print('Calling UNet')
                        points = self.unet_seg.predictContour(img)
                    else:
                        print('Rendering {0}'.format(img.shape))
                        img = enhance_contrast(img, disk(15))
                        image = img_as_float(img)
                        init_ls = checkerboard_level_set(image.shape, 3)
                        ls = morphological_chan_vese(image, 35, init_level_set=init_ls, smoothing=1)
                        points = list()
                        left_side, right_side = list(), list()
                        top_side, bottom_side = list(), list()
                        v_line = ls[ls.shape[0] // 2, :] * 255
                        dv_line = cv2.Scharr(v_line, ddepth=-1, dx=0, dy=1)
                        # dv_line = dv_line / dv_line.max()
                        try:
                            left, right = np.where(np.abs(dv_line) > 0)[0][0], np.where(np.abs(dv_line) > 0)[0][-1]
                        except Exception as e:
                            right, left = 0, 0
                            print(e)
                        # print(dv_line[left], dv_line[right])
                        # v = 0
                        if dv_line[left] < 0 and dv_line[right] > 0:
                            v = 1
                        elif dv_line[left] > 0 and dv_line[right] < 0:
                            v = 0
                        else:
                            self.statusBar().showMessage('Möglicherweise konnten nicht alle Konturen berechnet werden (1)')
                            self.statusBar().show()
                            continue
                        ls[0:5, :] = v
                        ls[-5:, :] = v
                        ls[:, 0:5] = v
                        ls[:, -5:] = v
                        for y in range(ls.shape[0]):
                            l = ls[y, :] * 255
                            try:
                                d_line = np.abs(cv2.Scharr(l, ddepth=-1, dx=0, dy=1))
                                if d_line.max() == 0:
                                    continue
                                d_line = (d_line / d_line.max()).astype(np.uint8)
                                x_left, x_right = np.where(d_line == 1)[0][0], np.where(d_line == 1)[0][-1]
                                left_side.append((y, x_left))
                                right_side.append((y, x_right))
                            except Exception as e:
                                continue
                        for x in range(ls.shape[1]):
                            l = ls[:, x] * 255
                            try:
                                d_line = np.abs(cv2.Scharr(l, ddepth=-1, dx=0, dy=1))
                                if d_line.max() == 0:
                                    continue
                                d_line = (d_line / d_line.max()).astype(np.uint8)
                                y_top, y_bottom = np.where(d_line == 1)[0][0], np.where(d_line == 1)[0][-1]
                                top_side.append((y_top, x))
                                bottom_side.append((y_bottom, x))
                            except Exception as e:
                                continue

                        # points = [x for x in left_side+list(reversed(right_side)) if x in top_side+list(reversed(bottom_side))]
                        points = [x for i, x in enumerate(left_side + list(reversed(right_side))) if (x in top_side + list(reversed(bottom_side)) and i % 3 == 0)]
                    if len(points) < 5:
                        self.canvas.shapes[i].contour_points = list()
                    else:
                        self.canvas.shapes[i].contour_points = points.copy()
        if rendered:
            self.saveFile()

    def renderSegmentationOverlay(self):
        # from OpenCV watershed example algorithm
        # print(self.image.width(), self.image.height())
        if self.canvas.shapes == []:
            return
        else:
            fullImg = cv2.imread(self.filePath, cv2.IMREAD_COLOR)
            # print(self.image.height(), self.image.width())
            segmentationOverlay = np.zeros((self.image.height(), self.image.width(), 3), dtype=np.uint8)
            # print(segmentationOverlay.shape)
            for s in self.canvas.shapes:
                xmin, ymin = int(s.points[0].x()), int(s.points[0].y())
                xmax, ymax = int(s.points[2].x()), int(s.points[2].y())
                img = fullImg[ymin:ymax+1, xmin:xmax+1, :]
                # print(segmentationOverlay[ymin:ymax+1, xmin:xmax+1, :].shape)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                if gray is None:
                    continue
                blur = cv2.GaussianBlur(gray, (3,3), 0)
                gray = cv2.addWeighted(gray, 1.5, blur, -0.8, 0)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                kernel = np.ones((3,3), np.uint8)
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                sure_bg = cv2.dilate(opening,kernel, iterations=1)
                dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
                _, sure_fg = cv2.threshold(dist_transform, 0.05 * dist_transform.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg, sure_fg)
                _, markers = cv2.connectedComponents(sure_fg)
                markers = markers + 1
                markers[unknown == 255] = 0
                markers = cv2.watershed(img, markers)
                segmentation = np.zeros(img.shape).astype(np.uint8)
                segmentation[markers == -1] = [0, 255, 0]
                segmentation[0:1, :] = segmentation.min()
                segmentation[-1:, :] = segmentation.min()
                segmentation[: , 0:1] = segmentation.min()
                segmentation[: ,-1:] = segmentation.min()
                segmentation = cv2.dilate(segmentation, np.ones((2,2)))
                # print(segmentation.shape, segmentationOverlay[ymin:ymax+1, xmin:xmax+1, :].shape, xmin, xmax, ymin, ymax)
                if segmentation.shape == segmentationOverlay[ymin:ymax+1, xmin:xmax+1, :].shape:
                    segmentationOverlay[ymin:ymax+1, xmin:xmax+1, :] = segmentation
                else:
                    continue
        # cv2.imwrite('segmentationOverlay.jpg', segmentationOverlay)
        return segmentationOverlay

    def renderOverlays(self):
        if self.showSegmentationOverlay:
            self.segmentationOverlay = self.renderSegmentationOverlay()
        else:
            self.segmentationOverlay = None

    def genOutput(self):
        if self.dirname is None: 
            return
        number_anno_files = len(glob.glob(self.dirname + '/' + '*.xml'))
        currentImg = io.imread(self.filePath)
        width, height = currentImg.shape[0], currentImg.shape[1]
        del(currentImg)
        dialog = scaleDialog(parent=self, width=width, height=height, scaling=self.pixel_scale)
        dialog.exec()
        self.pixel_scale = dialog.pixel_scale
        excel_filename = dialog.filename
        dialog.close()
        del(dialog)
        tableGenerator = cellTableGenerator(self.dirname + '/' + excel_filename + '.xlsx')
        progress = QProgressDialog('Berechne Ergebnisse {0}/{1}'.format(0, number_anno_files) , None, 0, 0, self)
        progress.setWindowTitle('Bitte warten')
        progress.setWindowModality(Qt.WindowModal)
        progress.setRange(0, number_anno_files)
        progress.setValue(0)
        progress.forceShow()
        for p in self.mImgList:
            marked_img_list = list()
            anno_file = p.split('.')[0] + '.xml'
            if not os.path.exists(anno_file):
                continue
            else:
                progress.setLabelText('Erzeuge Ergebnisse {0}/{1}'.format(progress.value() + 1, number_anno_files))
                progress.forceShow()
                self.filePath = p
                self.loadFile(self.filePath)
                draw_file = Image.open(self.filePath)
                draw = ImageDraw.Draw(draw_file)
                font = ImageFont.truetype('UbuntuMono.ttf', 30)
                image_filename = self.filePath.split('/')[-1]
                tableGenerator.add_cellcount(image_filename, len(self.canvas.shapes))
                for i, s in enumerate(self.canvas.shapes):
                    xmin, xmax, ymin, ymax = s.points[0].x(), s.points[2].x(), s.points[0].y(), s.points[2].y()
                    if not s.contour_points:
                        continue
                    else:
                        polygon_points = [(int(x+xmin), int(y+ymin)) for y, x in s.contour_points]
                        polygon = Polygon([(int(y), int(x)) for y, x in s.contour_points])
                        perimeter = polygon.length * self.pixel_scale  # polygon.length is defined as perimeter of polygon shape
                        r = perimeter / (2 * np.pi)  
                        V = (4/3) * np.pi * (r**3)
                        tableGenerator.add_cell(i+1, image_filename, polygon.area * (self.pixel_scale**2), perimeter, r, V)
                        draw.text((int(xmax - ((xmax - xmin)//2)), int(ymax - ((ymax - ymin)//2))), "{:.3f}".format(polygon.area * (self.pixel_scale**2)), fill=(0,0,0,255), font=font)
                        draw.text((int(xmin), int(ymin)), "{}".format(i+1), fill=(0,0,0,255), font=font)
                        draw.polygon(polygon_points, outline=(255,255,0,255))
                draw.text((10, 10), str(len(self.canvas.shapes)), fill=(0,0,0,255), font=font)
                draw_file.save(self.filePath.split('.')[0] + '_done' + '.jpg')
                progress.setValue(progress.value() + 1)
        progress.close()
        tableGenerator.close()
        info = QMessageBox.information(self, u'Information', 'Ergebnis wurde in {0}.xlsx gespeichert'.format(excel_filename))


    def loadRecent(self, filename, cellDetection=False):
        if not cellDetection:
            if self.dirty:
                self.saveFile()
        self.loadFile(filename)

    def scanAllImages(self, folderPath):
        extensions = ['.jpeg', '.jpg', '.png', '.bmp']
        images = []
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.join(root, file)
                    path = ustr(os.path.abspath(relativePath))
                    images.append(path)
        images.sort(key=lambda x: x.lower())
        return images

    def openDirDialog(self, _value=False, dirpath=None):
        if self.dirty:
            self.saveFile()
        defaultOpenDirPath = dirpath if dirpath else '.'
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'
        targetDirPath = ustr(QFileDialog.getExistingDirectory(self, '%s - Verzeichnis öffnen' % __appname__, defaultOpenDirPath, QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        self.importDirImages(targetDirPath)

    def importDirImages(self, dirpath):
        if not self.mayContinue() or not dirpath:
            return
        self.lastOpenDir = dirpath
        self.dirname = dirpath
        self.filePath = None
        self.fileListWidget.clear()
        self.mImgList = self.scanAllImages(dirpath)
        self.openNextImg()
        for imgPath in self.mImgList:
            item = QListWidgetItem(imgPath)
            self.fileListWidget.addItem(item)

    def openPrevImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving:
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            # else:
            #     self.changeSavedirDialog()
            #     return
        if self.dirty:
            self.saveFile()
        if len(self.mImgList) <= 0:
            return
        if self.filePath is None:
            return
        currIndex = self.mImgList.index(self.filePath)
        if currIndex - 1 >= 0:
            filename = self.mImgList[currIndex - 1]
            if filename:
                self.resetOverlays()
                self.loadFile(filename)
                if self.canvas.showContourOverlay:
                    self.calcContours()


    def openNextImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving:
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            # else:
            #     self.changeSavedirDialog()
            #     return
        if self.dirty:
            self.saveFile()
        if len(self.mImgList) <= 0:
            return
        filename = None
        if self.filePath is None:
            filename = self.mImgList[0]
        else:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex + 1 < len(self.mImgList):
                filename = self.mImgList[currIndex + 1]
        if filename:
            self.resetOverlays()
            self.loadFile(filename)
            if self.canvas.showContourOverlay:
                    self.calcContours()

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(ustr(self.filePath)) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image & Label files (%s)" % ' '.join(formats + ['*%s' % LabelFile.suffix])
        filename = QFileDialog.getOpenFileName(self, '%s - Choose Image or Label file' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.loadFile(filename)

    def saveFile(self, _value=False):
        imgFileDir = os.path.dirname(self.filePath)
        imgFileName = os.path.basename(self.filePath)
        savedFileName = os.path.splitext(imgFileName)[0] + XML_EXT
        savedPath = os.path.join(imgFileDir, savedFileName)
        self._saveFile(savedPath)

    def _saveFile(self, annotationFilePath):
        if annotationFilePath and self.saveLabels(annotationFilePath):
            self.setClean()


    def closeFile(self, _value=False):
        if self.dirty:
            self.saveFile()
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        #self.actions.saveAs.setEnabled(False)

    def resetAll(self):
        self.settings.reset()
        self.close()
        proc = QProcess()
        proc.startDetached(os.path.abspath(__file__))

    def mayContinue(self):
        return not (self.dirty and not self.discardChangesDialog())

    def discardChangesDialog(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'Sie haben ungespeicherte Annotationen. Dennoch fortfahren ?'
        return yes == QMessageBox.warning(self, u'Ungespeicherte Annotationen', msg, yes | no)

    def noAnnotationFileDialog(self):
        ok = QMessageBox.Ok
        msg = u'Keine Datei mit Annnotationen gefunden !'
        return QMessageBox.information(self, u'Keine Annotationen gefunden', msg, ok)

    def sureWithSavinDialog(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'Sind alle Annotationen und Konturen wie gewünscht ? \n Die alten gespeicherten Annotationen werden \nvollständig überschrieben'
        return yes == QMessageBox.warning(self, u'Speichern', msg, yes | no)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title, '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def deleteSelectedShape(self):
        self.toggleDrawMode(True)
        self.remLabel(self.canvas.deleteSelected())
        self.saveFile()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def loadPredefinedClasses(self, predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.labelHist is None:
                        self.labelHist = [line]
                    else:
                        self.labelHist.append(line)

    def loadPascalXMLByFilename(self, xmlPath):
        if self.filePath is None:
            return
        if os.path.isfile(xmlPath) is False:
            return

        tVocParseReader = PascalVocReader(xmlPath)
        shapes = tVocParseReader.getShapes()
        self.loadLabels(shapes)
        self.canvas.verified = tVocParseReader.verified


def read(filename, default=None):
    try:
        img = io.imread(filename)
        return img
    except:
        return default


def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app_icon = QIcon()
    app_icon.addFile('icon.png', QSize(225, 225))
    app.setWindowIcon(app_icon)
    win = MainWindow(argv[1] if len(argv) >= 2 else None,
                     argv[2] if len(argv) >= 3 else os.path.join(
                         os.path.dirname(sys.argv[0]),
                         'data', 'predefined_classes.txt'))
    win.show()
    return app, win


def main(argv=[]):
    '''construct main app and run it'''
    app, _win = get_main_app(argv)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
