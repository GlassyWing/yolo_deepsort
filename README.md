# yolo_deepsort

MOT base on yolo3/yolo4+deepsort, different from the official use of numpy to implement sort, the sort here reimplemented with pytorch, so it running at GPU.

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

## Replacing Re-ID

Please read the [tutorial](docs/how_to_replace_reid.md).

## Example

<img src="assets/track.gif">

## Training

This library does not contain a feasible training program, please refer to the training:

[yolo](https://github.com/AlexeyAB/darknet)

[deepsort](https://github.com/ZQPei/deep_sort_pytorch)

## References

https://github.com/eriklindernoren/PyTorch-YOLOv3

https://github.com/ZQPei/deep_sort_pytorch
