# Project #

c++ programs for object detection using
OpenCV with Tensorflow generated models

# Dependencies #

- [OpenCV](https://opencv.org) with the modules dnn, highgui, and imgproc installed
- TensorFlow configuration and model weight files (in resources folder); see [TensorFlow](https://github.com/tensorflow/models/tree/master/research/object_detection) object detection model repo for more information.
- C++17 compiler

# Installation #
From the github source with cmake
```bash
git clone https://github.com/whalenpt/OpenCVObjectDetect.git
cd OpenCVObjectDetect
cmake -S . -B build
cd build
cmake --build . -j4
```
# License #
- This project is licensed under the MIT License - see the [LICENSE](./LICENSE.txt) file for details.
- OpenCV has the [License](https://github.com/opencv/opencv/blob/master/LICENSE). 
- TensorFlow has the [License](//github.com/tensorflow/models/blob/master/LICENSE).

# Contact # 
Patrick Whalen - whalenpt@gmail.com

