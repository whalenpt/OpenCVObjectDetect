#include <opencv2/highgui.hpp> // cv::imshow(...), ...
#include <opencv2/dnn/dnn.hpp> // object detect functions
#include <opencv2/imgproc.hpp> // cv::rectangle(...)
#include <string>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <fstream>

const std::filesystem::path IMAGE_PATH = "resources/bike_person.jpg";
int main()
{
    // Files we need for object detection
    std::filesystem::path config_path = "resources/yolov4-tiny.cfg";
    std::filesystem::path model_path = "resources/yolov4-tiny.weights";
    std::filesystem::path label_path = "resources/coco_names.txt";

    if(!std::filesystem::exists(config_path))
        throw std::runtime_error("Could not find configuration .pbtxt file");
    if(!std::filesystem::exists(model_path))
        throw std::runtime_error("Could not find model weights .pb file");
    if( !std::filesystem::exists(label_path))
        throw std::runtime_error("Could not find label names file");

    // Load and set-up model
    cv::dnn::DetectionModel model(model_path.string(),config_path.string());
    int model_width = 416, model_height = 416;
    model.setInputSize(model_width,model_height);
    model.setInputScale(1.0/255);
    model.setInputMean(cv::Scalar(0.0,0.0,0.0));
    model.setInputCrop(false);

    // Load label names
    std::ifstream label_file(label_path);
    
    std::vector<std::string> label_names;
    std::string label_name;
    while(label_file >> label_name)
        label_names.push_back(label_name);
    if(label_names.empty())
        throw std::runtime_error("Failed to load any label names");

    // Load image to process
    if( !std::filesystem::exists(IMAGE_PATH))
        throw std::runtime_error("Could not find image file");
    cv::Mat img = cv::imread(IMAGE_PATH.string());

    // Detect and classify object
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    float conf_threshold = 0.25f; // need low threshold to detect focal person in bike_person.jpg
    model.detect(img,class_ids,confidences,boxes,conf_threshold);

    // Perform non-maximum suppression (of repeatedly detected objects)
    std::vector<int> indices;
    float nms_threshold = 0.5f; // higher -> keep more boxes
    cv::dnn::NMSBoxes(boxes,confidences,conf_threshold,nms_threshold,indices);

    // Draw boxes around detected objects and name them
    double font_scale = 1.25;
    int font_thickness = 1;
    int line_thickness = 2;

    // Loop over NMS indices
    for(auto i : indices) {
        cv::Rect& box = boxes[i];
        cv::rectangle(img,box,cv::Scalar(0,255,255),line_thickness);
        int class_id = class_ids[i];
        std::string label_name(label_names[class_id]);
        cv::putText(img,label_name,cv::Point(box.x+10,box.y+40),\
            cv::HersheyFonts::FONT_HERSHEY_TRIPLEX,font_scale,\
            cv::Scalar(255,255,0),font_thickness,cv::LineTypes::LINE_AA);
    }
    cv::namedWindow("Image",cv::WindowFlags::WINDOW_KEEPRATIO);
    cv::imshow("Image",img);
    cv::waitKey(0);
    return 0;
}
