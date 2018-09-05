#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <qstring.h>

using namespace cv;
using namespace std;

//our sensitivity value to be used in the absdiff() function
const static int SENSITIVITY_VALUE = 20;
//size of blur used to smooth the intensity image output from absdiff() function
const static int BLUR_SIZE = 10;
//we'll have just one object to search for
//and keep track of its position.
float theObject[2] = {0,0};
//bounding rectangle of the object, we will use the center of this as its position.
Rect objectBoundingRectangle = Rect(0,0,0,0);

//int to string helper function
string intToString(int number)
{
    //this function has a number input and string output
    std::stringstream ss;
    ss << number;
    return ss.str();
}

void onmouse(int event, int x, int y, int, void*)
{
    if( event == EVENT_LBUTTONDOWN )
    {
        theObject[0] = x;
        theObject[1] = y;
    }
}

void detect(QString videoPath)
{
    //some boolean variables for added functionality
 //   bool objectDetected = false;
    bool trackingEnabled = true;
    //pause and resume code
    bool pause = false;
    //set up the matrices that we will need
    //the two frames we will be comparing
    Mat frame1,frame2;
    //their grayscale images (needed for absdiff() function)
    Mat grayImage1,grayImage2;
    //resulting difference image
    Mat differenceImage;
    //thresholded difference image (for use in findContours() function)
    Mat thresholdImage;

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

    // Transition matrix ( eg. p(k) = p(k-1) + v(k-1)*dT ), init dT = 1
    double dT = 0.05;  // assume ~20fps

    ///////////
    /// \brief stdstrVideoPath
    ///
    string stdstrVideoPath = videoPath.toUtf8().constData();
    // Open video file
    VideoCapture video(stdstrVideoPath);
    namedWindow( "Display window", WINDOW_AUTOSIZE );

    //setMouseCallback("Display window", onmouse, 0);

    for(;;)
    {
        while(video.get(CAP_PROP_POS_FRAMES) < video.get(CAP_PROP_FRAME_COUNT)-1)
        {
            //read first frame
            video.read(frame1);
            //convert frame1 to gray scale for frame differencing
            cv::cvtColor(frame1,grayImage1,COLOR_BGR2GRAY);
            //copy second frame
            video.read(frame2);
            //convert frame2 to gray scale for frame differencing
            cv::cvtColor(frame2,grayImage2,COLOR_BGR2GRAY);
            //perform frame differencing with the sequential images. This will output an "intensity image"
            //do not confuse this with a threshold image, we will need to perform thresholding afterwards.
            cv::absdiff(grayImage1,grayImage2,differenceImage);
            //threshold intensity image at a given sensitivity value
            //cv::threshold(differenceImage,thresholdImage,SENSITIVITY_VALUE,255,THRESH_BINARY);

            Canny(grayImage1, grayImage1, 10, 150, 3);
            cv::threshold(grayImage1,thresholdImage,SENSITIVITY_VALUE,255,THRESH_BINARY);

            //blur the image to get rid of the noise. This will output an intensity image
            //cv::blur(thresholdImage,thresholdImage,cv::Size(BLUR_SIZE,BLUR_SIZE));
            //threshold again to obtain binary image from blur output
            //cv::threshold(thresholdImage,thresholdImage,SENSITIVITY_VALUE,255,THRESH_BINARY);
            //if tracking enabled, search for contours in our thresholded image
            imshow("thr", thresholdImage);

            cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));

            cv::dilate(thresholdImage, thresholdImage, structuringElement5x5);
            cv::dilate(thresholdImage, thresholdImage, structuringElement5x5);
            cv::erode(thresholdImage, thresholdImage, structuringElement5x5);

            if(trackingEnabled)
            {
                bool objectDetected = false;
                Mat temp;
                thresholdImage.copyTo(temp);
                //these two vectors needed for output of findContours
                vector< vector<Point> > contours;
                vector<Vec4i> hierarchy;
                //find contours of filtered image using openCV findContours function
                //findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );// retrieves all contours
                findContours(temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );// retrieves external contours

//                Canny(grayImage1, grayImage1, 10, 150, 3);
//                cv::threshold(grayImage1, grayImage1,SENSITIVITY_VALUE,255,THRESH_BINARY);
//                //imshow("tem", grayImage1);
//                findContours(grayImage1,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );


//                setMouseCallback("Display window", onmouse, 0);
//                cv::Point center;
//                center.x = theObject[0];
//                center.y = theObject[1];
//                cv::circle(frame1, center, 20, CV_RGB(255,0,0), -1);

                //if contours vector is not empty, we have found some objects
                if(contours.size() > 0)
                    objectDetected=true;
                else
                    objectDetected = false;

                if(objectDetected)
                {
                    vector<Rect> boundRect(contours.size());
                    vector<float>radius( contours.size() );
                    vector<Point2f>center( contours.size() );
                    RNG rng(12345);
                    vector<vector<Point> >hull( contours.size() );

                    for (size_t i = 0; i < contours.size(); i++)
                    {
                        convexHull( Mat(contours[i]), hull[i], false );

//                        boundRect[i] = boundingRect(Mat(contours[i]));
//                        minEnclosingCircle( (Mat)contours[i], center[i], radius[i] );
//                        drawContours(frame1, contours,  int (i), Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) ) );
                        drawContours(frame1, hull,  int (i), Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) ) );


//                        if (radius[i] > 10)
//                        {
//                            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
//                            rectangle( frame1, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
//                        }
                    }

                    //the largest contour is found at the end of the contours vector
                    //we will simply assume that the biggest contour is the object we are looking for.
//                    vector< vector<Point> > largestContourVec;
//                    largestContourVec.push_back(contours.at(contours.size()-1));
//                    //make a bounding rectangle around the largest contour then find its centroid
//                    //this will be the object's final estimated position.
//                    objectBoundingRectangle = boundingRect(largestContourVec.at(0));
//                    int xpos = objectBoundingRectangle.x + objectBoundingRectangle.width/2;
//                    int ypos = objectBoundingRectangle.y + objectBoundingRectangle.height/2;

//                    //update the objects positions by changing the 'theObject' array values
//                    theObject[0] = xpos , theObject[1] = ypos;

//                    kf.transitionMatrix.at<float>(2) = dT;
//                    kf.transitionMatrix.at<float>(9) = dT;

//                    state = kf.predict();
//                    cv::Rect predRect;
//                    predRect.width = state.at<float>(4);
//                    predRect.height = state.at<float>(5);
//                    predRect.x = state.at<float>(0) - predRect.width / 2;
//                    predRect.y = state.at<float>(1) - predRect.height / 2;

//                    cv::Point center;
//                    center.x = theObject[0];
//                    center.y = theObject[1];
//                    cv::circle(frame1, center, 2, CV_RGB(255,0,0), -1);
                    //cv::rectangle(frame1, predRect, CV_RGB(255,0,0), 2);
                }

                //make some temp x and y variables so we dont have to type out so much
//                int x = theObject[0];
//                int y = theObject[1];
                meas.at<float>(0) = theObject[0];
                meas.at<float>(1) = theObject[1];
                meas.at<float>(2) = (float)objectBoundingRectangle.width;
                meas.at<float>(3) = (float)objectBoundingRectangle.height;

                // >>>> Initialization
                kf.errorCovPre.at<float>(0) = 1; // px
                kf.errorCovPre.at<float>(7) = 1; // px
                kf.errorCovPre.at<float>(14) = 1;
                kf.errorCovPre.at<float>(21) = 1;
                kf.errorCovPre.at<float>(28) = 1; // px
                kf.errorCovPre.at<float>(35) = 1; // px

                state.at<float>(0) = meas.at<float>(0);
                state.at<float>(1) = meas.at<float>(1);
                state.at<float>(2) = 0;
                state.at<float>(3) = 0;
                state.at<float>(4) = meas.at<float>(2);
                state.at<float>(5) = meas.at<float>(3);
                // <<<< Initialization

                kf.statePost = state;


                //draw some crosshairs around the object
               // circle(frame1,Point(state.at<float>(0), state.at<float>(1)), 40,Scalar(0,255,0),2);

                Rect etm;
                etm.width = state.at<float>(4);
                etm.height = state.at<float>(5);
                etm.x = state.at<float>(0) - etm.width / 2;
                etm.y = state.at<float>(1) - etm.height / 2;
                rectangle(frame1, etm, Scalar(0,255,0), 1, CV_AA, 0);

                //write the position of the object to the screen
                putText(frame1,"Tracking object at (" + intToString(state.at<float>(0))+","+intToString(state.at<float>(1))+")",Point(state.at<float>(0), state.at<float>(1)),1,1,Scalar(255,0,0),2);
            }
            //show our captured frame
            //namedWindow( "Display window", WINDOW_AUTOSIZE );
            imshow("Display window",frame1);
            switch(waitKey(100)){
            case 116: //'t' has been pressed. this will toggle tracking
                trackingEnabled = !trackingEnabled;
                if(trackingEnabled == false) cout<<"Tracking disabled."<<endl;
                else cout<<"Tracking enabled."<<endl;
                break;
            case 112: //'p' has been pressed. this will pause/resume the code.
                pause = !pause;
                if(pause == true){ cout<<"Code paused, press 'p' again to resume"<<endl;
                while (pause == true){
                    //stay in this loop until
                    switch (waitKey()){
                        //a switch statement inside a switch statement? Mind blown.
                    case 112:
                        //change pause back to false
                        pause = false;
                        cout<<"Code Resumed"<<endl;
                        break;
                    }
                }
                }



            }
//            int key = (char)waitKey(60);
//            if(key == 32)
//                break;
        }




//        Mat rawFrame;
//        video >> rawFrame;

//        Mat gray_frame;
//        cvtColor(rawFrame, gray_frame, CV_BGR2GRAY);

//        Ptr<BackgroundSubtractorMOG2> pMOG2;
//        Mat fgMaskMOG2;
//        pMOG2 = createBackgroundSubtractorMOG2();
//        pMOG2->apply(gray_frame, fgMaskMOG2);

//        Canny(gray_frame, gray_frame, 0, 255, 3);
//        Mat dst_frame;
//        threshold(gray_frame, dst_frame, 20, 255, 0);

//        RNG rng(12345);

//        vector<vector<Point> > contours;
//        vector<Vec4i> hierarchy;

//        findContours( dst_frame, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//        int blob_radius_thresh = 60;

//        /// Approximate contours to polygons + get bounding rects and circles
//        vector<vector<Point> > contours_poly( contours.size() );
//        vector<Rect> boundRect( contours.size() );
//        vector<Point2f>center( contours.size() );
//        vector<float>radius( contours.size() );

//        for( int i = 0; i < contours.size(); i++ )
//        {
//            approxPolyDP( Mat(contours[i]), contours_poly[i], 1, true );
//            boundRect[i] = boundingRect( Mat(contours_poly[i]) );
//            minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );

//             ///
//             if (radius[i] > blob_radius_thresh)
//             {
//                 Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
//                 rectangle( rawFrame, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
//                 imshow("Kaka", rawFrame);

//             }
//        }

//        // Stop video when press "space"
//        int key = (char)waitKey(1);
//        if(key == 32)
//            break;
//    }

    // Detele the video capture object
    video.release();
    // Close all frame
    destroyAllWindows();
    }
}

void searchForMovement(Mat thresholdImage, Mat &cameraFeed)
{
    //notice how we use the '&' operator for objectDetected and cameraFeed. This is because we wish
    //to take the values passed into the function and manipulate them, rather than just working with a copy.
    //eg. we draw to the cameraFeed to be displayed in the main() function.
    bool objectDetected = false;
    Mat temp;
    thresholdImage.copyTo(temp);
    //these two vectors needed for output of findContours
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    //find contours of filtered image using openCV findContours function
    //findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );// retrieves all contours
    findContours(temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );// retrieves external contours

    //if contours vector is not empty, we have found some objects
    if(contours.size()>0)objectDetected=true;
    else objectDetected = false;

    if(objectDetected)
    {
        //the largest contour is found at the end of the contours vector
        //we will simply assume that the biggest contour is the object we are looking for.
        vector< vector<Point> > largestContourVec;
        largestContourVec.push_back(contours.at(contours.size()-1));
        //make a bounding rectangle around the largest contour then find its centroid
        //this will be the object's final estimated position.
        objectBoundingRectangle = boundingRect(largestContourVec.at(0));
        int xpos = objectBoundingRectangle.x+objectBoundingRectangle.width/2;
        int ypos = objectBoundingRectangle.y+objectBoundingRectangle.height/2;

        //update the objects positions by changing the 'theObject' array values
        theObject[0] = xpos , theObject[1] = ypos;
    }
    //make some temp x and y variables so we dont have to type out so much
    int x = theObject[0];
    int y = theObject[1];

    //draw some crosshairs around the object
    circle(cameraFeed,Point(x,y),20,Scalar(0,255,0),2);
    line(cameraFeed,Point(x,y),Point(x,y-25),Scalar(0,255,0),2);
    line(cameraFeed,Point(x,y),Point(x,y+25),Scalar(0,255,0),2);
    line(cameraFeed,Point(x,y),Point(x-25,y),Scalar(0,255,0),2);
    line(cameraFeed,Point(x,y),Point(x+25,y),Scalar(0,255,0),2);

    //write the position of the object to the screen
    putText(cameraFeed,"Tracking object at (" + intToString(x)+","+intToString(y)+")",Point(x,y),1,1,Scalar(255,0,0),2);
}
