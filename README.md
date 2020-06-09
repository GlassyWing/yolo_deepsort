# yolo_deepsort

基于yolo3/yolo4+deepsort的多目标追踪

## 主要依赖

```
pytorch >= 1.3
torchvision >= 0.4.0
```

## 快速开始

1. 克隆项目

```sh
git clone https://github.com/GlassyWing/yolo_deepsort
```

2. 下载yolo3或者yolo4模型文件

```sh
cd weights/
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
wget https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view
```

3. 下载deepsort模型文件

```sh
# download ckpt.t7 from
https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 to this folder
```

4. 运行

```sh
python video_deepsort.py
```
若不想运行跟踪器，将参数`tracker`设为`None`即可：

```
video_detector = VideoDetector(...
                               tracker=None)
```

## 示例

<img src="assets/track.gif">

## 训练

本仓库不包含可行的训练程序，训练请参考：

[yolo训练参考](https://github.com/AlexeyAB/darknet)

[deepsort训练参考](https://github.com/ZQPei/deep_sort_pytorch)

## 引用

https://github.com/eriklindernoren/PyTorch-YOLOv3

https://github.com/ZQPei/deep_sort_pytorch
