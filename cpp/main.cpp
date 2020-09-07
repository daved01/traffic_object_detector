#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
//#include <stdio.h>

using namespace cv;
using namespace std;

int main()
{
    // Open video from webcam
    Mat frame;
    // ---- INITIALIZE VIDEOCAPTURE AND VIDEOWRITER
    VideoCapture cap;
    // open default camera
    cap.open(0);
    // check if camera has been opened
    if (!cap.isOpened() ) {
        cerr << "ERROR! Cannot open camera\n";
        return -1;
    }
    /*
    // Set up video output
    VideoWriter out;
    string pathout = "./output/output.avi";
    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
    Size s = Size((int) 640, (int) 360);
    //Size s = Size((int) cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));
    //const double fps = 10.0;
    const double fps = cap.get(CAP_PROP_FPS);
    out.open(pathout, codec, fps, s);
    // check if we succeeded
    if (!out.isOpened()) {
        cerr << "Could not open the output video file for write\n";
        return -1;
    }
    */

    // Loop through each frame
    for (;;)
    {
        // Grab new frame
        cap.read(frame);
        // Check if successful
        if (frame.empty() ) {
            cerr << "Cannot receive frame (stream end?). Exiting ...\n";
            break;
        }

        /* For selected frame find bounding boxes here
        ...
        */

       // Write frame to video
       //out.write(frame);


       // Show images
       imshow("Object Detection Demo", frame);
       if (waitKey(5) >= 0)
        break;
    }
    
    
    return 0;
}