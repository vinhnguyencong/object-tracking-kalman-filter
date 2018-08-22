#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <qstring.h>

using namespace cv;
using namespace std;

//our sensitivity value to be used in the absdiff() function
const static int SENSITIVITY_VALUE = 50;
//size of blur used to smooth the intensity image output from absdiff() function
const static int BLUR_SIZE = 10;
//we'll have just one object to search for
//and keep track of its position.
int theObject[2] = {0,0};
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

    ///////////
    /// \brief stdstrVideoPath
    ///
    string stdstrVideoPath = videoPath.toUtf8().constData();
    // Open video file
    VideoCapture video(stdstrVideoPath);

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
            cv::threshold(differenceImage,thresholdImage,SENSITIVITY_VALUE,255,THRESH_BINARY);
            //blur the image to get rid of the noise. This will output an intensity image
            cv::blur(thresholdImage,thresholdImage,cv::Size(BLUR_SIZE,BLUR_SIZE));
            //threshold again to obtain binary image from blur output
            cv::threshold(thresholdImage,thresholdImage,SENSITIVITY_VALUE,255,THRESH_BINARY);
            //if tracking enabled, search for contours in our thresholded image
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

                //if contours vector is not empty, we have found some objects
                if(contours.size() > 0)objectDetected=true;
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
                    int xpos = objectBoundingRectangle.x + objectBoundingRectangle.width/2;
                    int ypos = objectBoundingRectangle.y + objectBoundingRectangle.height/2;

                    //update the objects positions by changing the 'theObject' array values
                    theObject[0] = xpos , theObject[1] = ypos;
                }
                //make some temp x and y variables so we dont have to type out so much
                int x = theObject[0];
                int y = theObject[1];

                //draw some crosshairs around the object
                circle(frame1,Point(x,y),40,Scalar(0,255,0),2);
//                line(frame1,Point(x,y),Point(x,y-25),Scalar(0,255,0),2);
//                line(frame1,Point(x,y),Point(x,y+25),Scalar(0,255,0),2);
//                line(frame1,Point(x,y),Point(x-25,y),Scalar(0,255,0),2);
//                line(frame1,Point(x,y),Point(x+25,y),Scalar(0,255,0),2);

                //write the position of the object to the screen
                putText(frame1,"Tracking object at (" + intToString(x)+","+intToString(y)+")",Point(x,y),1,1,Scalar(255,0,0),2);
            }
            //show our captured frame
            imshow("Frame1",frame1);
            switch(waitKey(10)){
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
