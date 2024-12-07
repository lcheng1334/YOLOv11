import cv2
from ultralytics import YOLO

# 加载YOLOv11模型
model = YOLO("/home/lc/code/yolo/yolo11n-seg.pt")
model.eval()

cap = cv2.VideoCapture(0)
cap.set(3, 300)
cap.set(4, 100)

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
    print(boxes)

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