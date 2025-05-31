# import cv2
# from ultralytics import YOLO
#
# # 1. 加载 YOLO 模型（使用 YOLOv5 或者 YOLOv8）
# model = YOLO(r"D:\daima\python\srp\runs\train\exp\weights\best.pt")  # 你可以替换为适合的模型，支持 YOLOv8，YOLOv5 等
#
# # 2. 读取图像
# image_path = r'D:\daima\python\srp\rps\test\images\20220216_221856_jpg.rf.c551cb3856f480cba36d6aa58e3300cd.jpg'  # 替换为你的图像路径
# image = cv2.imread(image_path)
#
# # 获取原始图像的尺寸
# original_height, original_width = image.shape[:2]
#
# # 3. 将图像调整为 416x416
# resized_image = cv2.resize(image, (416, 416))
#
# # 4. 使用 YOLO 模型进行推理
# results = model(resized_image)
#
# # 获取模型标记结果（bounding boxes）
# boxes = results[0].boxes  # 获取检测框
# labels = results[0].names  # 获取标签名称
#
# # 5. 在 YOLO 处理后的图像上绘制检测框
# for box in boxes:
#     x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取坐标
#     confidence = box.conf[0].item()  # 获取置信度
#     label = labels[int(box.cls[0].item())]  # 获取标签
#
#     # 绘制框和标签
#     cv2.rectangle(resized_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#     cv2.putText(resized_image, f'{label} {confidence:.2f}', (x1, y1-10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
# # 6. 恢复图像至原始尺寸
# restored_image = cv2.resize(resized_image, (original_width, original_height))
#
# # 7. 显示处理后的图像
# cv2.imshow("Processed Image", restored_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 8. 保存处理后的图像（如果需要）
# cv2.imwrite("processed_image.jpg", restored_image)
# print(label)


import cv2
from ultralytics import YOLO

# 1. 加载模型
model = YOLO(r"D:\daima\python\srp\runs\train\exp\weights\best.pt")

# 2. 读取并调整图像
image_path = r'D:\daima\python\srp\rps\test\images\egohands-public-1620852256502_png_jpg.rf.b7a5cc4785712f79c89a9ff3ca7c883e.jpg'
image = cv2.imread(image_path)
original_height, original_width = image.shape[:2]
resized_image = cv2.resize(image, (416, 416))

# 3. 推理与结果处理
results = model(resized_image)
boxes = results[0].boxes
labels = results[0].names

# 4. 遍历检测结果并打印
for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    confidence = box.conf[0].item()
    label = labels[int(box.cls[0].item())]

    # 打印每个检测结果
    print(f"目标: {label}, 置信度: {confidence:.2f}")

    # 绘制检测框
    cv2.rectangle(resized_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(resized_image, f'{label} {confidence:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# 5. 恢复尺寸并显示
restored_image = cv2.resize(resized_image, (original_width, original_height))
cv2.imshow("Processed Image", restored_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. 保存结果（可选）
cv2.imwrite("processed_image.jpg", restored_image)