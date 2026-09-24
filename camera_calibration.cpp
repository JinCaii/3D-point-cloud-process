#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

int main()
{
    //定义棋盘格的尺寸和角点数量
    Size boardSize(9, 6); // 棋盘格的内角点数量
    float squareSize = 0.025f; // 棋盘格每个方格的边长，单位为米

    vector<vector<Point3f>> objectPoints;  //棋盘格的真实三维点坐标
    vector<vector<Point2f>> imagePoints;   //角点在照片中的像素坐标

    //生成所有内角点的真实坐标（x,y,0），并依次存进objp
    vector<Point3f> objp;  //一张棋盘格的所有内角点的三维坐标
    for (int i = 0; i < boardSize.height; i++)  //6行
    {
        for (int j = 0; j < boardSize.width; j++)  //9列
        {
            objp.push_back(Point3f(j * squareSize, i * squareSize, 0)); //共54个Point3f点,push_back()依次往vector<>objp中添加元素
        }
    }

    //读取棋盘格图片
    vector<String> images;
    glob("chessboard/*.jpg", images); //读取chessboard文件夹下所有jpg图片
    Mat gray; //转换为灰度图，查找角点的函数只支持灰度图
    vector<Point2f> corners; //存储每一张图片角点的像素坐标
    
    //遍历每一张图片，查找角点
    for (size_t i = 0; i < images.size(); i++)
    {
        Mat img = imread(images[i]); //读取图片
        cvtColor(img, gray, COLOR_BGR2GRAY); //转换为灰度图
        bool found = findChessboardCorners(gray, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE); //查找角点
        //若找到了棋盘格角点
        if (found)
        {
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.001)); //精确化角点位置
            imagePoints.push_back(corners); //将每张图片的角点像素坐标存入imagePoints
            objectPoints.push_back(objp); //将每张图片的真实三维点坐标存入objectPoints
            Mat show = img.clone(); //克隆一张图片用于显示

            drawChessboardCorners(img, boardSize, corners, found); //在图片上绘制角点
            imshow("Chessboard", img); //显示图片
            waitKey(500); //等待500ms
        }
        
    }

    //标定相机
    Mat cameraMatrix, distCoeffs; //相机内参矩阵和畸变系数
    vector<Mat> rvecs, tvecs; //旋转向量和平移向量，外参
    double error = calibrateCamera(objectPoints, imagePoints, gray.size(), cameraMatrix, distCoeffs, rvecs, tvecs); //标定相机

    cout << "error: " << error << endl; //输出重投影误差
    cout << "cameraMatrix: " << cameraMatrix << endl; //输出相机内参矩阵    
    cout << "distCoeffs: " << distCoeffs << endl; //输出畸变系数
    return 0;
}