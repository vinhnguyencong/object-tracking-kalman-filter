#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <qstring.h>

using namespace cv;
using namespace std;

// Sensitivity value
const static int SENS_VAL = 20;

// Convert int to string
string intToString(int number)
{
    std::stringstream ss;
    ss << number;
    return ss.str();
}

// Get coordinate of mouse when left click
//void onMouse(int event, int x, int y, int, void*)
//{
//    if( event == EVENT_LBUTTONDOWN )
//    {
//        xposition[0] = x;
//        xposition[1] = y;
//    }
//}

void detect(QString videoPath)
{
    //some boolean variables for added functionality
 //   bool objectDetected = false;
    bool trackingEnabled = true;
    //pause and resume code
    bool pause = false;

    // Kalman Filter
    int stateSize = 6;
    int measSize = 4;
    int contrSize = 0;


    unsigned int type = CV_32F;
    KalmanFilter kf(stateSize, measSize, contrSize, type);

    Mat state(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
    Mat meas(measSize, 1, type);    // [z_x,z_y,z_w,z_h]

    //Mat procNoise(stateSize, 1, type);
    // [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

    // Transition State Matrix A
    // Note: set dT at each processing step!
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]
    cv::setIdentity(kf.transitionMatrix);

    // Measure Matrix H
    // [ 1 0 0 0 0 0 ]
    // [ 0 1 0 0 0 0 ]
    // [ 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 1 ]
    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;

    // Process Noise Covariance Matrix Q
    // [ Ex   0   0     0     0    0  ]
    // [ 0    Ey  0     0     0    0  ]
    // [ 0    0   Ev_x  0     0    0  ]
    // [ 0    0   0     Ev_y  0    0  ]
    // [ 0    0   0     0     Ew   0  ]
    // [ 0    0   0     0     0    Eh ]
    //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 5.0f;
    kf.processNoiseCov.at<float>(21) = 5.0f;
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;

    // Measures Noise Covariance Matrix R
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
    // End Kalman Filter

    // Brief stdstrVideoPath
    string stdstrVideoPath = videoPath.toUtf8().constData();

    // Capture video
    VideoCapture video(stdstrVideoPath);
    namedWindow( "Display window", WINDOW_AUTOSIZE );

    // Transition matrix ( eg. p(k) = p(k-1) + v(k-1)*dT )
    double fps = video.get(CV_CAP_PROP_FPS);

    // dT is time between 2 frame
    double dT = 1/fps;

    Mat rawFrame, grayFrame, thresholdFrame;

    for(;;)
    {
        while(video.get(CAP_PROP_POS_FRAMES) < video.get(CAP_PROP_FRAME_COUNT)-1)
        {
            // Read video frame
            video.read(rawFrame);

            // Convert RGB frame to gray
            cv::cvtColor(rawFrame,grayFrame,COLOR_BGR2GRAY);

            // Find edge line of object
            Canny(grayFrame, grayFrame, 10, 150, 3);
            cv::threshold(grayFrame,thresholdFrame,SENS_VAL,255,THRESH_BINARY);

            //threshold again to obtain binary image from blur output
            cv::threshold(thresholdFrame,thresholdFrame,SENS_VAL,255,THRESH_BINARY);

            // Return rectangular structuring element with size 5x5
            cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

            cv::dilate(thresholdFrame, thresholdFrame, structuringElement5x5);
            cv::dilate(thresholdFrame, thresholdFrame, structuringElement5x5);
            cv::erode(thresholdFrame, thresholdFrame, structuringElement5x5);

            //and keep track of its position.
            float theObject[2] = {0,0};

            if(trackingEnabled)
            {
                bool objectDetected = false;
                Mat temp;
                thresholdFrame.copyTo(temp);
                //these two vectors needed for output of findContours
                vector< vector<Point> > contours;
                vector<Vec4i> hierarchy;

                // Find boundary of onject
                findContours(temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );

                // Handle left mouse click
//                setMouseCallback("Display window", onMouse, 0);


                // Found object with the number of contours > 0
                if(contours.size() > 0)
                {
                    objectDetected=true;
                }
                else
                {
                    objectDetected = false;
                }

                vector<Rect> objectBoundRect(contours.size());
                vector<vector<Point> > hull( contours.size() );
                vector<Rect> predictRectangle(contours.size());

                if(objectDetected)
                {
                    for (size_t i = 0; i < contours.size(); i++)
                    {
                        // Find convex hull of contours
                        convexHull( Mat(contours[i]), hull[i], false );
                        objectBoundRect[i] = boundingRect(Mat(hull[i]));
                        //rectangle( rawFrame, objectBoundRect[i].tl(), objectBoundRect[i].br(), CV_RGB(0,0,255), 2, 8, 0 );

                        kf.transitionMatrix.at<float>(2) = dT;
                        kf.transitionMatrix.at<float>(9) = dT;

                        state = kf.predict();

                        predictRectangle[i].width = state.at<float>(4);
                        predictRectangle[i].height = state.at<float>(5);
                        predictRectangle[i].x = state.at<float>(0) - predictRectangle[i].width / 2;
                        predictRectangle[i].y = state.at<float>(1) - predictRectangle[i].height / 2;
                        rectangle(rawFrame, predictRectangle[i], CV_RGB(255,0,0), 2);
                    }

                }

//                for (size_t i = 0; i < contours.size(); i++)
//                {
//                    meas.at<float>(0) = ;
//                    meas.at<float>(1) = ;
//                    meas.at<float>(2) = (float)objectBoundRect.width;
//                    meas.at<float>(3) = (float)objectBoundRect.height;

//                    // >>>> Initialization
//                    kf.errorCovPre.at<float>(0) = 1; // px
//                    kf.errorCovPre.at<float>(7) = 1; // px
//                    kf.errorCovPre.at<float>(14) = 1;
//                    kf.errorCovPre.at<float>(21) = 1;
//                    kf.errorCovPre.at<float>(28) = 1; // px
//                    kf.errorCovPre.at<float>(35) = 1; // px

//                    state.at<float>(0) = meas.at<float>(0);
//                    state.at<float>(1) = meas.at<float>(1);
//                    state.at<float>(2) = 0;
//                    state.at<float>(3) = 0;
//                    state.at<float>(4) = meas.at<float>(2);
//                    state.at<float>(5) = meas.at<float>(3);
//                    // <<<< Initialization

//                    kf.statePost = state;

//                    Rect etm;
//                    etm.width = state.at<float>(4);
//                    etm.height = state.at<float>(5);
//                    etm.x = state.at<float>(0) - etm.width / 2;
//                    etm.y = state.at<float>(1) - etm.height / 2;
//                    rectangle(rawFrame, etm, Scalar(0,255,0), 1, CV_AA, 0);

//                    //write the position of the object to the screen
//                    putText(rawFrame,"Tracking object at (" + intToString(state.at<float>(0))+","+intToString(state.at<float>(1))+")",Point(state.at<float>(0), state.at<float>(1)),1,1,Scalar(255,0,0),2);
//                }

            //show our captured frame
            //namedWindow( "Display window", WINDOW_AUTOSIZE );
            imshow("Display window",rawFrame);

            switch(waitKey(60))
            {
            // Pause or Resume by press Space
            case 32:
                pause = !pause;
                if(pause == true)
                {
                    while (pause == true)
                    {
                        //stay in this loop until
                        switch (waitKey(60))
                        {
                        case 32:
                            // Resume
                            pause = false;
                            break;
                        }
                    }
                }
            }

            // Handle pause and resume -- End

        }

    }

    // Detele the video capture object
    video.release();

    // Close all windows frame
    destroyAllWindows();
    }
}
