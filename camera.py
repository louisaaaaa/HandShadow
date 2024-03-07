import cv2
import torch
import tkinter as tk
from PIL import Image, ImageTk
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import numpy as np


def adjust_gamma(image, gamma=1.0):
    # Set up Look-up-Table for gamma correction
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # Apply Gamma Correction
    return cv2.LUT(image, table)

# 加载模型
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)  # 假设您有3个类别
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# 定义转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 类别
classes = ['deer', 'snake','bird','dog','other']

# 初始化摄像头
cap = cv2.VideoCapture(0)

def update_frame():
    # 捕获一帧图像
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return
    
    '''
    # Histogram Equalize 
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Gamma correction, suppose gamma=1.2
    frame = adjust_gamma(frame, gamma=1.2)
    
    # Denoising
    frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
'''
    # color space conversion
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    
    # 将捕获的帧转换为Tkinter格式
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # 转换图像以适应模型输入
    img_for_pred = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_t = transform(img_for_pred)
    batch_t = torch.unsqueeze(img_t, 0)

    # 进行预测
    out = model(batch_t)
    _, index = torch.max(out, 1)

    # 更新预测结果
    result_var.set(f'Prediction: {classes[index[0]]}')
    
    # 每50毫秒刷新一次界面
    root.after(50, update_frame)

# 创建UI
root = tk.Tk()
root.title("Real-time Image Classification and Video Stream")

# 用于显示视频帧的Label
video_label = tk.Label(root)
video_label.pack()

# 用于显示预测结果的Label
result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, font=('Helvetica', 20))
result_label.pack()

# 开始更新帧
update_frame()

# 启动UI循环
root.mainloop()

# 释放资源
cap.release()
cv2.destroyAllWindows()
