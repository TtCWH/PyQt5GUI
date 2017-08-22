from ChooseUI import Ui_Dialog
from PyQt5 import QtCore, QtGui, QtWidgets
import shutil
import os.path
import Parameters

class ChooseRightLabel(Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, wrong_image, label_list, parent=None):
        super(ChooseRightLabel, self).__init__()
        self.setupUi(self)

        self.wrong_image = wrong_image
        for label in label_list:
            self.comboBox.addItem(label)
        self.show()

        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.accepted.connect(self.add_img2db)

    def add_img2db(self):
        dest_path = os.path.join(Parameters.TRAININGDATABASE ,Parameters.MODEL_SAVE_NAME, self.comboBox.currentText())
        shutil.copy(self.wrong_image, dest_path)
        QtWidgets.QMessageBox.information(self, 'Thanks', 'Thank you for the feedback.\nI will be better!')
        os.remove(self.wrong_image)









