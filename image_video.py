import cv2

# 加载视频
video_path = '/home/lc/code/yolo/runs/segment/predict/How to Train Ultralytics YOLOv8 models on Your Custom Dataset in Google Colab ｜ Episode 3 [LNwODJXcvt4].avi'  # 这里替换为你的视频文件路径
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)

# 计算每帧之间的等待时间，单位是毫秒
wait_time = int(1000 / fps)

while True:
    # 逐帧读取视频
    ret, frame = cap.read()
    
    if not ret:
        print("视频读取完毕")
        break
    
    # 显示每一帧
    cv2.imshow('Frame', frame)
    
    # 按 'q' 键退出
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

# 释放视频捕捉对象并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
