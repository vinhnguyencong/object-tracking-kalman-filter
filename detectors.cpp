#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <qstring.h>
#include <detectors.h>

using namespace cv;
using namespace std;

void detect(QString videoPath)
{
    string stdstrVideoPath = videoPath.toUtf8().constData();
    // Open video file
    VideoCapture video(stdstrVideoPath);
    for(;;){
    Mat rawFrame;
    video >> rawFrame;

    Mat gray_frame;
    cvtColor(rawFrame, gray_frame, CV_BGR2GRAY);

    Ptr<BackgroundSubtractorMOG2> pMOG2;
    Mat fgMaskMOG2;
    pMOG2 = createBackgroundSubtractorMOG2();
    pMOG2->apply(gray_frame, fgMaskMOG2);

    Canny(gray_frame, gray_frame, 50, 190, 3);
    Mat dst_frame;
    threshold(gray_frame, dst_frame, 240, 255, 0);

    RNG rng(12345);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours( dst_frame, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    int blob_radius_thresh = 60;

    /// Approximate contours to polygons + get bounding rects and circles
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );

    for( int i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );

         ///
         if (radius[i] > blob_radius_thresh)
         {
             circle(rawFrame, center[i], radius[i], (0, 255, 0), 2);
             imshow("Kaka", rawFrame);

         }
    }

    // Stop video when press "space"
    int key = (char)waitKey(100);
    if(key == 32)
        break;
    }
}

