#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

#include <memory>
#include <torch/script.h>
//#include <stdio.h>

using namespace cv;
using namespace std;

int main()
{
    // Variables
    Size s = Size((int) 640, (int) 360);
    int frame_num = 10;
    // File containing the object detection model being imported via torchscript (*.pt)
    const char* model_file = "/Users/David/Repositories/traffic_object_detector/cpp/build/model/fasterrcnn50fpn.pt";
    // Import object detection model model from .pt file
    torch::jit::script::Module module;
    
    try {
        torch::jit::getProfilingMode() = false;   
        torch::jit::getExecutorMode() = false;
        torch::jit::setGraphExecutorOptimize(false);
        module = torch::jit::load(model_file);
        std::cout << "Model import success!\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "Error in loading the model\n";
        return -1;
    }


    // Open video from webcam
    Mat frame; // Create header part
    // ---- INITIALIZE VIDEOCAPTURE AND VIDEOWRITER
    VideoCapture cap;
    // open default camera
    cap.open(0);
    // check if camera has been opened
    if (!cap.isOpened() ) {
        cerr << "ERROR! Cannot open camera\n";
        return -1;
    }
    
    // Set up video output
    VideoWriter out;
    string pathout = "/Users/David/Repositories/traffic_object_detector/cpp/build/output/output.avi";
    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
    
    //Size s = Size((int) cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));
    //const double fps = 10.0;
    const double fps = cap.get(CAP_PROP_FPS);
    out.open(pathout,codec,fps, s);
    // check if we succeeded
    if (!out.isOpened()) {
        cerr << "Could not open the output video file for write\n";
        return -1;
    }


    // Loop through each frame
    int counter = 0;
    for (;;)
    {
        // Grab new frame
        cap.read(frame);     
        // Check if successful
        if (frame.empty() ) {
            cerr << "Cannot receive frame (stream end?). Exiting ...\n";
            break;
        }
        resize(frame, frame, s);

        // For selected frame find bounding boxes here
        if (counter % frame_num == 0)
        {
            cout << "Frame number..." << endl;
       
            Mat frame_rgb;
            // Swap axis from BGR to RGB
            cvtColor(frame, frame_rgb, COLOR_BGR2RGB); // Convert bgr to rgb
            // Convert to tensor and add batch dimension
            at::Tensor tensor_image = torch::from_blob(frame_rgb.data, {1, 3, frame_rgb.rows, frame_rgb.cols}, at::kByte);
            cout << "tensor_image = " << endl << " " << tensor_image << endl << endl;
            // normalize
              
            // Execute the model and turn its output into a tensor.
            auto output = module.forward(tensor_image).toTensor();
            Mat output_mat(cv::Size(640,360), CV_8UC3, output.data<float>());

            Mat output8, output_bgr;
            cvtColor(output8, output_bgr, COLOR_RGB2BGR);


            //Mat output_mat({1920, 1080}, CV_8UC3, output.data<uint8>());
            
            
            // Bounding box prediction
            
        



            // Show images
            imshow("Object Detection Demo", frame);
            // Write frame to video
            out.write(frame);

            counter = 0;
        }
        counter++;

        if (waitKey(5) >= 0)
            break;
    }
    
    // Release video capture and write object
    cap.release();
    out.release();

    // Close all windows
    destroyAllWindows();

    return 0;
}