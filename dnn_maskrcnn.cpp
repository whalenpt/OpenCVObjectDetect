#include <opencv2/highgui.hpp> // cv::imshow(...), ...
#include <opencv2/dnn/dnn.hpp> // object detect functions
#include <opencv2/imgproc.hpp> // cv::rectangle(...)
#include <random>
#include <string>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <fstream>

const std::filesystem::path IMAGE_PATH = "resources/bike_person.jpg";
int main()
{
    // Files we need for object detection
    std::filesystem::path config_path = "resources/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
    std::filesystem::path model_path = "resources/mask_rcnn_inception_v2_coco_2018_01_28_weights.pb";
    // Need to update label_path (rcnn model has 90 class ids, not 80)
    std::filesystem::path label_path = "resources/coco_names.txt";

    if(!std::filesystem::exists(config_path))
        throw std::runtime_error("Could not find configuration .pbtxt file");
    if(!std::filesystem::exists(model_path))
        throw std::runtime_error("Could not find model weights .pb file");
    if( !std::filesystem::exists(label_path))
        throw std::runtime_error("Could not find label names file");

    // Load image to process
    if( !std::filesystem::exists(IMAGE_PATH))
        throw std::runtime_error("Could not find image file");
    cv::Mat img = cv::imread(IMAGE_PATH.string());
    int height = img.size[0], width = img.size[1];

    // Load and set-up model
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(model_path.string(),config_path.string());
    cv::Mat blob; // Create 4D Mat tensor -> blob
    cv::dnn::blobFromImage(img,blob,1.0,cv::Size(),cv::Scalar(),true);
    net.setInput(blob);
    std::vector<cv::Mat> output;
    std::vector<std::string> output_names = {"detection_out_final","detection_masks"};
    net.forward(output,output_names);

    // Load label names
    std::ifstream label_file(label_path);
    std::vector<std::string> label_names;
    std::string label_name;
    while(label_file >> label_name)
        label_names.push_back(label_name);
    if(label_names.empty())
        throw std::runtime_error("Failed to load any label names");

    // Segment objects
    cv::Mat allboxes = output[0].reshape(1,output[0].total()/7);
    cv::Mat allmasks = output[1];
    const cv::Size mask_size(allmasks.size[2],allmasks.size[3]);

    float conf_threshold = 0.6f;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Mat> masks;
    for(int i = 0; i < allboxes.size[0]; i++){
        float confidence = allboxes.at<float>(i, 2);
        if (confidence > conf_threshold){
            int class_id = static_cast<int>(allboxes.at<float>(i, 1));
            class_ids.push_back(class_id);
            confidences.push_back(confidence);
            int x = static_cast<int>(width * allboxes.at<float>(i, 3));
            int y = static_cast<int>(height * allboxes.at<float>(i, 4));
            int x2 = static_cast<int>(width * allboxes.at<float>(i, 5));
            int y2 = static_cast<int>(height * allboxes.at<float>(i, 6));
            boxes.emplace_back(cv::Point2i(x,y),cv::Point2i(x2,y2));
            masks.emplace_back(mask_size,CV_32FC1,allmasks.ptr<float>(i,class_id));
        }
    }

    // Perform non-maximum suppression (of repeatedly detected objects)
    std::vector<int> indices;
    float nms_threshold = 0.35f; // higher -> keep more boxes
    cv::dnn::NMSBoxes(boxes,confidences,conf_threshold,nms_threshold,indices);

    // Create colors for each class label
    std::vector<cv::Scalar> colors(label_names.size());
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_int_distribution<int> distr(0,255);
    for(int i = 0; i < label_names.size(); i++)
        colors[i] = cv::Scalar(distr(engine),distr(engine),distr(engine));

    // Draw boxes around detected objects and name them
    double font_scale = 1.25;
    int font_thickness = 1;
    int line_thickness = 2;
    int contour_thickness = 2;
    double mask_threshold = 0.4;

    for(auto i : indices) {
        // Draw the bounding box
        cv::Rect& box = boxes[i];
        cv::rectangle(img,box,cv::Scalar(0,255,255),line_thickness);

        // Label the bounding box
        int class_id = class_ids[i];
        std::string label_name(label_names[class_id]);
        cv::putText(img,label_name,cv::Point(box.x+10,box.y+40),\
            cv::HersheyFonts::FONT_HERSHEY_TRIPLEX,font_scale,\
            cv::Scalar(255,255,0),font_thickness,cv::LineTypes::LINE_AA);

        // Draw the segmentation mask
        cv::Scalar color = colors[class_id];
        // Region of interest (current object box/add color)
        cv::Mat roi = 0.4*color + 0.6*img(box);

        // Process mask
        cv::Mat& mask = masks[i];
        cv::resize(mask,mask,cv::Size(box.width, box.height));
        cv::threshold(mask,mask,mask_threshold,255,cv::THRESH_BINARY);
        mask.convertTo(mask,CV_8UC1);

        std::vector<cv::Mat> contours;
        cv::Mat hierarchy;
        cv::findContours(mask,contours,hierarchy,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_TC89_KCOS);
        cv::drawContours(roi,contours,-1,color,contour_thickness,cv::LINE_AA,hierarchy);
        roi.copyTo(img(box),mask);
    }

    cv::namedWindow("Image",cv::WindowFlags::WINDOW_KEEPRATIO);
    cv::imshow("Image",img);
    cv::waitKey(0);
    return 0;
}


