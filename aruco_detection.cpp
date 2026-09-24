#include <Zivid/Zivid.h>
#include <Zivid/Experimental/Calibration.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <iostream>
#include <vector>

int main()
try
{
    // 定义ArUco字典和参数
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
    cv::aruco::DetectorParameters parameters;
    cv::aruco::ArucoDetector detector(dictionary, parameters);

    // Application 必须在连接和采集期间保持存活（Zivid SDK 2.11）。
    Zivid::Application zivid;
    std::cout << "Connecting to Zivid camera..." << std::endl;
    auto camera = zivid.connectCamera();
    std::cout << "Connected: " << camera.info() << std::endl;

    // 使用 SDK 默认的单次 2D 曝光设置。
    const Zivid::Settings2D settings{
        Zivid::Settings2D::Acquisitions{Zivid::Settings2D::Acquisition{}}};

    // 按当前 2D 采集设置获取匹配图像分辨率的内参。
    const auto intrinsics = Zivid::Experimental::Calibration::intrinsics(camera, settings);
    const auto &matrix = intrinsics.cameraMatrix();
    const auto &distortion = intrinsics.distortion();
    const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
        matrix.fx().value(), 0, matrix.cx().value(),
        0, matrix.fy().value(), matrix.cy().value(),
        0, 0, 1);
    const cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) <<
        distortion.k1().value(), distortion.k2().value(),
        distortion.p1().value(), distortion.p2().value(), distortion.k3().value());
    const std::string windowName = "ArUco Detection";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 800, 600);

    while (true)
    {
        // SDK 2.11 使用 capture(Settings2D) 采集二维图像。
        const auto capturedFrame = camera.capture(settings);
        const auto image = capturedFrame.imageBGRA();
        const cv::Mat bgra(static_cast<int>(image.height()),
                           static_cast<int>(image.width()), CV_8UC4,
                           const_cast<Zivid::ColorBGRA *>(image.data()));
        // cvtColor 生成独立的 BGR 图像；不修改 SDK 所有的 BGRA 数据。
        cv::Mat frame;
        cv::cvtColor(bgra, frame, cv::COLOR_BGRA2BGR);
        // 转换为灰度图像，非必要
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // 检测ArUco二维码
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
        detector.detectMarkers(gray, markerCorners, markerIds, rejectedCandidates);
        // 也可以直接传入彩色 frame，省略前面的灰度转换。

        // 绘制检测结果
        if (markerIds.size() > 0)
        {
            cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

            // 估计标记相对于相机的位姿。边长单位为米，tvec 的单位也为米。
            const float arucoLength = 0.03f;
            const float halfLength = arucoLength / 2.0f;
            // 标记中心为原点，按左上、右上、右下、左下排列。
            // 此顺序与检测角点及 SOLVEPNP_IPPE_SQUARE 的要求一致。
            const std::vector<cv::Point3f> objectPoints = {
                {-halfLength,  halfLength, 0.0f},
                { halfLength,  halfLength, 0.0f},
                { halfLength, -halfLength, 0.0f},
                {-halfLength, -halfLength, 0.0f}
            };

            for (size_t i = 0; i < markerIds.size(); ++i)
            {
                cv::Vec3d rvec, tvec;
                const bool poseFound = cv::solvePnP(
                    objectPoints, markerCorners[i], cameraMatrix, distCoeffs,
                    rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
                if (!poseFound)
                    continue;

                std::cout << "Marker ID: " << markerIds[i] << std::endl;
                std::cout << "rvec: " << rvec << std::endl;
                std::cout << "tvec (m): " << tvec << std::endl;
                cv::drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.1f);
            }
        }

        // 无论是否找到标记，都显示检测状态。
        cv::putText(frame, "DICT_4X4_250 | Detected: " + std::to_string(markerIds.size()),
                    cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, "Q / Esc / close window to exit", cv::Point(10, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

        // 显示图像
        cv::imshow(windowName, frame);

        // 处理窗口事件后，检查按键和关闭按钮。
        const int key = cv::waitKey(10);
        if (key == 'q' || key == 'Q' || key == 27 ||
            cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1)
        {
            break;
        }
    }

    // 释放资源
    camera.disconnect();
    cv::destroyAllWindows();

    return 0;
}
catch (const std::exception &e)
{
    std::cerr << "Zivid/OpenCV error: " << Zivid::toString(e) << std::endl;
    std::cerr << "Check camera power/network and disconnect it in Zivid Studio before retrying."
              << std::endl;
    cv::destroyAllWindows();
    return EXIT_FAILURE;
}
