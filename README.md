# Project #

c++ programs for object detection and image segmentation using the
OpenCV dnn module. Example programs use MobileNet, YOLO-tiny, and MaskRCNN neural networks
trained on the COCO dataset.

# Dependencies #

- [OpenCV](https://opencv.org) with the modules dnn, highgui, and imgproc installed
- Configuration and model weight files (in resources folder)
- C++17 compiler

# Build Examples #
From the github source with cmake
```bash
git clone https://github.com/whalenpt/OpenCVObjectDetect.git
cd OpenCVObjectDetect
cmake -S . -B build
cd build
cmake --build . -j4
```
Executables are named:
- mobilenet_image
- yolo_image
- maskrcnn_image

# License #
- This project is licensed under the MIT License - see the [LICENSE](./LICENSE.txt) file for details.
- OpenCV has the [License](https://github.com/opencv/opencv/blob/master/LICENSE). 
- TensorFlow has the [License](//github.com/tensorflow/models/blob/master/LICENSE).

# Contact # 
Patrick Whalen - whalenpt@gmail.com

