
# YOLOv4-Tiny Pothole Detection Pipeline

This guide details the steps to train a YOLOv4-tiny model for pothole detection, convert it to TensorFlow Lite format, and test it using sample images.

---

## 1. Load an Existing Model

Start by creating a new notebook in Google Colab and mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

Download the pre-trained YOLOv4-tiny weights:
```bash
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
```

---

## 2. Dataset Preparation

Download the pothole dataset from [Roboflow](https://public.roboflow.com/object-detection/pothole/1) in YOLO Darknet format.

- Extract `.zip` into a folder named `Images`.
- Delete any `README` and `.labels` files from `train`, `valid`, and `test` folders.
- Zip the cleaned `Images` folder as `obj.zip`.

---

## 3. Setup Required Files

Create a folder `yolov4-tiny` in your Google Drive, then a subfolder `training`.

Upload these to `yolov4-tiny`:
- `obj.zip`
- `yolov4-tiny-custom.cfg` (with modified training parameters)
- `obj.names`
- `obj.data`
- `process.py`

Modify `yolov4-tiny-custom.cfg`:
- `batch=64`, `subdivisions=16`, `width=416`, `height=416`
- `max_batches=6000`, `steps=4800,5400`
- Set correct `filters` and `classes`

---

## 4. Setup Darknet

Clone and configure Darknet:
```bash
!git clone https://github.com/AlexeyAB/darknet
%cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
!make
```

Link Drive:
```bash
!ln -s /content/drive/My\ Drive/ /mydrive
```

Prepare folders and copy config:
```bash
%cd data/
!find -maxdepth 1 -type f -exec rm -rf {{}} \;
%cd ..
%rm -rf cfg/
%mkdir cfg
!cp /mydrive/yolov4-tiny/obj.zip ../
!unzip ../obj.zip -d data/
!cp /mydrive/yolov4-tiny/yolov4-tiny-custom.cfg ./cfg
!cp /mydrive/yolov4-tiny/obj.names ./data
!cp /mydrive/yolov4-tiny/obj.data  ./data
!cp /mydrive/yolov4-tiny/process.py ./
!python process.py
```

---

## 5. Train the Model

Run training with transfer learning:
```bash
!./darknet detector train data/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -dont_show -map
```

Test the detector:
```bash
!wget -O /mydrive/yolov4-tiny/ph.jpg https://raw.githubusercontent.com/SanaulMalik/SherlockHoles/master/images/ph.jpg
!./darknet detector test data/obj.data cfg/yolov4-tiny-custom.cfg /mydrive/yolov4-tiny/training/yolov4-tiny-custom_best.weights /mydrive/yolov4-tiny/ph.jpg -thresh 0.3
```

---

## 6. Convert to `.pb` Format

Switch runtime from GPU to None.

Clone converter repo:
```bash
!git clone https://github.com/hunglc007/tensorflow-yolov4-tflite.git
```

Edit `core/config.py`:
```python
__C.YOLO.CLASSES = "mydrive/yolov4-tiny/obj.names"
```

Convert weights:
```bash
!python save_model.py --weights /mydrive/yolov4-tiny/training/yolov4-tiny-custom_best.weights --output /mydrive/yolov4-tiny/yolov4-tiny-pb --input_size 416 --model yolov4 --framework tflite --tiny
```

Verify:
```python
import tensorflow as tf
model = tf.keras.models.load_model("/mydrive/yolov4-tiny/yolov4-tiny-pb")
model.summary()
```

---

## 7. Convert to `.tflite`

Edit `convert_tflite.py`:
```python
converter.experimental_enable_resource_variables = True
```

Convert:
```bash
!python convert_tflite.py --weights /mydrive/yolov4-tiny/yolov4-tiny-pb --output /mydrive/yolov4-tiny/model.tflite --quantize_mode float16
```

---

## 8. Test the TFLite Model

```bash
!git clone https://github.com/SanaulMalik/SherlockHoles
%cd /mydrive/yolov4-tiny
!cp /content/SherlockHoles/executables/detect_tflite.py ./
!cp /content/SherlockHoles/images/ph-2.jpg ./ph-2.jpg
!python detect_tflite.py --weights model.tflite --size 416 --model yolov4 --image ph-2.jpg --framework tflite --tiny
```

---

## ✅ Output

The output will be saved as `result.png` in your current working directory (i.e. `yolov4-tiny` folder).

---

## 🧵 Acknowledgements

- [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
- [hunglc007/tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
- [Roboflow Pothole Dataset](https://public.roboflow.com/object-detection/pothole/1)
