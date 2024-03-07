import sys
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont
from PyQt5.QtCore import Qt, QTimer
import numpy as np


def adjust_gamma(image, gamma=1.0):
    # Set up Look-up-Table for gamma correction
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # Apply Gamma Correction
    return cv2.LUT(image, table)


class HandShadowApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hand Shadow")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Set background color
        self.central_widget.setStyleSheet("background-color: #4A708B;")

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Add instruction text
        self.instruction_label = QLabel("Please position your hands in front of the camera as shown as the hand shadow poses below!")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(18)
        self.instruction_label.setFont(font)
        self.instruction_label.setStyleSheet("color: white;")
        self.layout.addWidget(self.instruction_label)

        # Label to show camera flow
        self.camera_label = QLabel(self)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.camera_label)

        # Label to show detected hand pose
        self.hand_pose_label = QLabel(self)
        self.hand_pose_label.setAlignment(Qt.AlignCenter)
        font.setPointSize(25)
        self.hand_pose_label.setFont(font)
        self.hand_pose_label.setStyleSheet("color: Orange;")
        self.layout.addWidget(self.hand_pose_label)

        # Load Model
        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 6)
        self.model.load_state_dict(torch.load('model_weights.pth'))
        self.model.eval()

        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Class labels
        self.classes = ['bird', 'deer', 'dog', 'other', 'snake']

        # Set up camera
        self.cap = cv2.VideoCapture(0)

        # Timer for camera updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)  # Update every 50 milliseconds

        # Display photos
        self.display_photos()

    def display_photos(self):
        hbox = QHBoxLayout()
        self.layout.addLayout(hbox)

        labels = ['Snake', 'Deer', 'Bird', 'Dog']

        # Create a horizontal layout for each photo with its label
        for i, label_text in enumerate(labels):
            vbox = QVBoxLayout()  # Use QVBoxLayout for each photo and label pair
            vbox.setSpacing(0)  # Set spacing between widgets
            vbox.setAlignment(Qt.AlignCenter)  # Center align the contents of the layout
            hbox.addLayout(vbox)
  
            hbox.addLayout(vbox)
            font = QFont()
            font.setBold(True)
            font.setPointSize(15)
            # Add label
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignHCenter)  # Center align the label text
            label.setFont(font)
            label.setStyleSheet("color: white;")
            vbox.addWidget(label)

            # Add photo
            pixmap = QPixmap(f'photo{i+1}.jpg')  # Replace with your photo paths
            if not pixmap.isNull():
                pixmap = pixmap.scaledToWidth(150)
                photo_label = QLabel()
                photo_label.setPixmap(pixmap)
                vbox.addWidget(photo_label)
            else:
                print(f"Could not load photo{i+1}.jpg")

    def update_frame(self):
        # Capture a frame
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return

        ''' DSP processing part START'''
        # # Histogram Equalize
        # img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        # frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        # Gamma correction, suppose gamma=1.2
        # frame = adjust_gamma(frame, gamma=1.2)

        # # Denoising
        # frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

  

        # color space conversion
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ''' DSP processing part END'''
        
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.camera_label.setPixmap(pixmap.scaled(600, 450, Qt.KeepAspectRatio))  # Adjust size here

        # Convert the image to fit the model input
        img_for_pred = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_t = self.transform(img_for_pred)
        batch_t = torch.unsqueeze(img_t, 0)

        # Perform prediction
        out = self.model(batch_t)
        _, index = torch.max(out, 1)

        # Update the prediction result
        self.hand_pose_label.setText(f'Are you posing as : {self.classes[index[0]]}')

    def closeEvent(self, event):
        # Release resources when closing the window
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandShadowApp()
    window.show()
    sys.exit(app.exec_())
