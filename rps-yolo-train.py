import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO



if __name__ == '__main__':
    model=YOLO(r"D:\daima\python\srp\yolo11.yaml")
    model.train(data=r"D:\daima\python\srp\rps\data.yaml",
                cache=False,
                imgsz=640,
                epochs=1,
                single_cls=False,  # 是否是单类别检测
                batch=16,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD',
                amp=True,
                project='runs/train',
                name='exp',
                )