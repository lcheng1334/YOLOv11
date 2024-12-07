# -*-coding:utf-8 -*-
# Time: 2024/12/07
# Author: lcheng1334
# File: image_video.py
from ultralytics import YOLO

# 加载模型
model = YOLO("/home/lc/code/yolo/yolo11n-seg.pt")

# 文件视频检测
model.predict(source="/home/lc/code/yolo/input/Musk.png", show=True)

# 摄像头检测
model.predict(source="0", show=True)