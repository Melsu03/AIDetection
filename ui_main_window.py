# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\qtd_ui\ui\MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets

class QtMainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(561, 605)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(parent=self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 541, 291))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.wgtCamera = QtWidgets.QWidget(parent=self.verticalLayoutWidget)
        self.wgtCamera.setObjectName("wgtCamera")
        self.verticalLayout.addWidget(self.wgtCamera)
        self.btnCapture = QtWidgets.QPushButton(parent=self.centralwidget)
        self.btnCapture.setGeometry(QtCore.QRect(10, 340, 541, 41))
        self.btnCapture.setDefault(False)
        self.btnCapture.setFlat(False)
        self.btnCapture.setObjectName("btnCapture")
        self.lineEdit = QtWidgets.QLineEdit(parent=self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(10, 310, 481, 21))
        self.lineEdit.setObjectName("lineEdit")
        self.btnFromFile = QtWidgets.QPushButton(parent=self.centralwidget)
        self.btnFromFile.setGeometry(QtCore.QRect(500, 310, 51, 21))
        self.btnFromFile.setDefault(False)
        self.btnFromFile.setFlat(False)
        self.btnFromFile.setObjectName("btnFromFile")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AI Detection"))
        self.btnCapture.setText(_translate("MainWindow", "Capture from Camera"))
        self.btnFromFile.setText(_translate("MainWindow", "File"))
