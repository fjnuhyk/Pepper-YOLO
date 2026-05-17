import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO(r'Pepper-Yolo.yaml')

    model.train(data=r'PepperV6.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=4,

                optimizer='SGD',




                project='runs/train',
                name='exp',
                )
