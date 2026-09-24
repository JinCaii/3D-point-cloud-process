#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

int main()
{
    //定义棋盘格的尺寸和角点数量
    // Define the size of the chessboard and the number of inner corners
    Size boardSize(9, 6); // 棋盘格的内角点数量 The number of inner corners of the chessboard
    float squareSize = 0.025f; // 棋盘格每个方格的边长，单位为米  The length of each square of the chessboard, in meters

    vector<vector<Point3f>> objectPoints;  //棋盘格的真实三维点坐标 The real 3D coordinates of the chessboard
    vector<vector<Point2f>> imagePoints;   //角点在照片中的像素坐标 The pixel coordinates of the corners in the photos

    //生成所有内角点的真实坐标（x,y,0），并依次存进objp
    // Generate the real coordinates of all inner corners (x,y,0) and store them in objp in order
    vector<Point3f> objp;  //一张棋盘格的所有内角点的三维坐标  The 3D coordinates of all inner corners of a chessboard
    for (int i = 0; i < boardSize.height; i++)  //6行 6 rows
    {
        for (int j = 0; j < boardSize.width; j++)  //9列 9 columns
        {
            objp.push_back(Point3f(j * squareSize, i * squareSize, 0)); //共54个Point3f点,push_back()依次往vector<>objp中添加元素 A total of 54 Point3f points, push_back() adds elements to vector<>objp in order
        }
    }

    //读取棋盘格图片
    // Read chessboard images
    vector<String> images;
    glob("chessboard/*.jpg", images); //读取chessboard文件夹下所有jpg图片  Read all jpg images in the chessboard folder
    Mat gray; //转换为灰度图，查找角点的函数只支持灰度图 Convert to grayscale, the function for finding corners only supports grayscale images
    vector<Point2f> corners; //存储每一张图片角点的像素坐标 Store the pixel coordinates of the corners of each image
    
    //遍历每一张图片，查找角点
    // Traverse each image and find corners
    for (size_t i = 0; i < images.size(); i++)
    {
        Mat img = imread(images[i]); //读取图片 Read image
        cvtColor(img, gray, COLOR_BGR2GRAY); //转换为灰度图 Convert to grayscale
        bool found = findChessboardCorners(gray, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE); //查找角点 Find corners
        //若找到了棋盘格角点
        // If the chessboard corners are found
        if (found)
        {
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.001)); //亚像素优化 subpixel corner refinement
            imagePoints.push_back(corners); //将每张图片的角点像素坐标存入imagePoints  Store the pixel coordinates of the corners of each image in imagePoints
            objectPoints.push_back(objp); //将每张图片的真实三维点坐标存入objectPoints  Store the real 3D coordinates of each image in objectPoints
            Mat show = img.clone(); //克隆一张图片用于显示 Clone an image for display

            drawChessboardCorners(img, boardSize, corners, found); //在图片上绘制角点 Draw corners on the image
            imshow("Chessboard", img); //显示图片 Show image
            waitKey(500); //等待500ms Wait for 500ms
        }
        
    }

    //标定相机
    // Camera calibration
    Mat cameraMatrix, distCoeffs; //相机内参矩阵和畸变系数  Camera intrinsic matrix and distortion coefficients
    vector<Mat> rvecs, tvecs; //旋转向量和平移向量，外参 Rotation vectors and translation vectors, extrinsic parameters
    double error = calibrateCamera(objectPoints, imagePoints, gray.size(), cameraMatrix, distCoeffs, rvecs, tvecs); //标定相机 Camera calibration

    cout << "error: " << error << endl; //输出重投影误差 Output reprojection error
    cout << "cameraMatrix: " << cameraMatrix << endl; //输出相机内参矩阵 Output camera intrinsic matrix  
    cout << "distCoeffs: " << distCoeffs << endl; //输出畸变系数 Output distortion coefficients
    return 0;
}