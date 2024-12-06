# YOLOv11模型搭建
连接ubuntu系统后通过搭建环境已经完成目标检测模型。具体其他YOLOv11预训练可点击[链接](https://docs.ultralytics.com/models/yolo11/#key-features)，包括分类、目标检测等等。输入命令：
```bash
pip install ultralytics
```
同时安装好torch后，可以开始检测，例如我们使用yolo11n-seg.pt进行目标检测，在终端输入
```bash
yolo predict model=yolo11n-seg.pt source='/path/to/video'
```
source路径为你要检测的视频或者图像，运行后便可输出predict文件夹，得到检测结果。
效果非常好，如果还有更轻量化和更高准确率的模型可提供，我可以进行搭建并检测。
# 简单实现摄像头检测
根据前者所介绍的YOLOv11，我选择yolo11n-seg.pt作为检测的模型，在完成模型搭建后，可以直接使用该预训练模型通过摄像头检测从而得到识别的坐标、宽度、高度、置信度和类别，只需要通过cv2可视化即可。具体代码如下：

## 加载模型
```python
import cv2
from ultralytics import YOLO

# 加载YOLOv11模型
model = YOLO("/home/lc/code/yolo/yolo11n-seg.pt")
model.eval()
```

## 检测摄像头
```python
cap = cv2.VideoCapture(0)
```

# 检查摄像头是否成功打开
```python
if not cap.isOpened():
    print("无法打开摄像头")
    exit()
```

## 对识别物体进行框选
通过摄像头的每一帧图像放入模型中进行推理，返回的结果为字典型，我们提取其中的坐标、宽度、高度、置信度和类别即可。输出结果如下:
```text
cls: tensor([0.])
conf: tensor([0.8900])
data: tensor([[ 98.2722, 137.0070, 575.2213, 479.0414,   0.8900,   0.0000]])
id: None
is_track: False
orig_shape: (480, 640)
shape: torch.Size([1, 6])
xywh: tensor([[336.7467, 308.0242, 476.9490, 342.0344]])
xywhn: tensor([[0.5262, 0.6417, 0.7452, 0.7126]])
xyxy: tensor([[ 98.2722, 137.0070, 575.2213, 479.0414]])
xyxyn: tensor([[0.1536, 0.2854, 0.8988, 0.9980]])
```
选择我们需要的信息，使用**cv2.rectangle**绘图和**cv2.putText**显示类别比和置信度，最后在摄像头中显示结果。

## 总结
该运行结果基本上可以实现摄像头检测。（功能确实有，因为是直接运用模型检测的，但我总感觉过于简单，等待优化）
全部代码如下：
```python
import cv2
from ultralytics import YOLO

# 加载YOLOv11模型
model = YOLO("/home/lc/code/yolo/yolo11n-seg.pt")
model.eval()

cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 捕获摄像头的一帧图像
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头帧")
        break

    # 将图像传入YOLO模型进行推理
    results = model(frame)  # 进行YOLO推理

    # 获取边界框数据
    boxes = results[0].boxes  # 获取预测的边界框

    # 获取边界框坐标 (x, y, w, h)
    for box in boxes:  # 访问每个框
        # 获取坐标、宽度、高度、置信度和类别
        x, y, w, h = box.xywh[0]  # 提取坐标 (x_center, y_center, w, h)
        conf = box.conf[0]  # 获取置信度
        cls = box.cls[0]  # 获取类别索引
        # 过滤掉置信度低于0.5的框
        if conf > 0.5:
            # 计算边界框的坐标
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制类别标签和置信度
            cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示检测结果
    cv2.imshow("YOLOv11 Detection", frame)

    # 按 'q' 键退出程序
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
```

[预训练模型点击此处下载](https://github.com/lcheng1334/AI/blob/main/yolo11n-seg.pt)
[整体]
