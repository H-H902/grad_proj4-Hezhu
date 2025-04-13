import sys
import os
import torch
import torchaudio
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QLabel, QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QProgressBar
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, Qt, QThread, pyqtSignal
from src.models import MANNER as MANNER_BASE
from src.models_small import MANNER as MANNER_SMALL

class AudioEnhancer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = self.load_model('manner_base .pth')
        self.input_file = None
        self.output_file = None

        font = QFont('Times New Roman', 12)  # 设置字体为 Arial，大小为 12
        font_2 = QFont('Times New Roman', 10)  # 设置字体为 Arial，大小为 12

    def initUI(self):
        self.setWindowTitle('Audio Enhancer')
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        font = QFont('Times New Roman', 12)
        font_2 = QFont('Times New Roman', 10)

        self.label = QLabel('Audio Enhancer By He Zhu', self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(font)
        layout.addWidget(self.label)

        self.btn_select = QPushButton('Choose An Audio', self)
        self.btn_select.setFont(font_2)
        self.btn_select.clicked.connect(self.openFileNameDialog)
        layout.addWidget(self.btn_select)

        self.btn_confirm = QPushButton('Process', self)
        self.btn_confirm.setFont(font_2)
        self.btn_confirm.clicked.connect(self.startProcessing)
        layout.addWidget(self.btn_confirm)

        self.progress = QProgressBar(self)
        layout.addWidget(self.progress)

        self.status_label = QLabel('', self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(font)
        layout.addWidget(self.status_label)

        self.btn_play_input = QPushButton('Play the Input Audio', self)
        self.btn_play_input.setFont(font_2)
        self.btn_play_input.clicked.connect(self.playInputAudio)
        layout.addWidget(self.btn_play_input)

        self.btn_play_output = QPushButton('Play the Output Audio', self)
        self.btn_play_output.setFont(font_2)
        self.btn_play_output.clicked.connect(self.playOutputAudio)
        layout.addWidget(self.btn_play_output)

        self.setLayout(layout)

        self.player = QMediaPlayer()
    def openFileNameDialog(self):
      options = QFileDialog.Options()
      options |= QFileDialog.ReadOnly
      fileName, _ = QFileDialog.getOpenFileName(self, "选择噪声文件", "", "音频文件 (*.wav);;所有文件 (*)",
                                              options=options)
      if fileName:
        self.input_file = fileName
        self.label.setText(f'Input Files: {fileName}')

    def load_model(self, model_name):
        model = MANNER_BASE(in_channels=1, out_channels=1, hidden=60, depth=4, kernel_size=8, stride=4, growth=2, head=1, segment_len=64).to('cuda:0')
        checkpoint = torch.load(f'C:\\Users\\sagacious h\\Pycharmprojects\\MANNER\\weights\\{model_name}')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

    def startProcessing(self):
            if self.input_file:
                self.status_label.setText('Processing...')
                self.progress.setValue(0)
                self.thread = ProcessingThread(self.input_file, self.model)
                self.thread.progress.connect(self.updateProgress)
                self.thread.finished.connect(self.processingFinished)
                self.thread.start()

    def updateProgress(self, value):
            self.progress.setValue(value)

    def processingFinished(self, output_file):
            self.output_file = output_file
            self.status_label.setText(f'Complete!: {output_file}')

    def playInputAudio(self):
            if self.input_file:
                self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.input_file)))
                self.player.play()

    def playOutputAudio(self):
            if self.output_file:
                self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.output_file)))
                self.player.play()

class ProcessingThread(QThread):
        progress = pyqtSignal(int)
        finished = pyqtSignal(str)

        def __init__(self, input_file, model):
            super().__init__()
            self.input_file = input_file
            self.model = model

        def run(self):
            output_file = self.process_audio(self.input_file)
            self.finished.emit(output_file)

        def process_audio(self, input_file):
            output_file = input_file.replace('.wav', '_enhanced.wav')
            noisy, sr = torchaudio.load(input_file)
            if sr != 16000:
                tf = torchaudio.transforms.Resample(sr, 16000)
                noisy = tf(noisy)

            noisy = noisy.unsqueeze(0).to('cuda:0')
            with torch.no_grad():
                enhanced = self.model(noisy)
            enhanced = enhanced.squeeze(0).detach().cpu()
            torchaudio.save(output_file, enhanced, 16000)
            self.progress.emit(100)
            return output_file

if __name__ == '__main__':
        app = QApplication(sys.argv)
        ex = AudioEnhancer()
        ex.show()
        sys.exit(app.exec_())