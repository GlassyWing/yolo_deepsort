# yolo_deepsort

MOT base on yolo3/yolo4+deepsort

## Mainly dependent

```
pytorch >= 1.3
torchvision >= 0.4.0
opencv-python >= 4.1
```

## Quick start

1. Clone the repositorty

```sh
git clone https://github.com/GlassyWing/yolo_deepsort
```

2. Download weights of yolo3 or yolo4

```sh
cd weights/
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
wget https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view
```

3. Download weights of DeepSort

```sh
# download ckpt.t7 from
https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 to this folder
```

4. Run example

```sh
python video_deepsort.py
```
If you do not want to run the tracker, set the parameter `tracker` to `None`:

```
video_detector = VideoDetector(...
                               tracker=None)
```

## Example

<img src="assets/track.gif">

## Tests

| GPU    | MODEL                   | Predict time | FPS |
| ------ | ----------------------- | ------------ | --- |
| 1070Ti | YOLOv3-608              | 45ms         | 18  |
| 1070Ti | YOLOv3-608-DeepSort     | 70ms         | 12  |
| 1070Ti | YOLOv4-608              | 64ms         | 13  |
| 1070Ti | YOLOv4-608-DeepSort     | 88ms         | 10  |
| 1070Ti | YOLOv4-608-F16          | 52ms         | 16  |
| 1070Ti | YOLOv4-608-F16-DeepSort | 74ms         | 11  |

There is a bottleneck in the opencv python version, that is to draw boxs, which will cause great IO consumption

## Training

This library does not contain a feasible training program, please refer to the training:

[yolo](https://github.com/AlexeyAB/darknet)

[deepsort](https://github.com/ZQPei/deep_sort_pytorch)

## References

https://github.com/eriklindernoren/PyTorch-YOLOv3

https://github.com/ZQPei/deep_sort_pytorch
