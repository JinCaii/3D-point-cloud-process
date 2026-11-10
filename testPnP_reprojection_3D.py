# ============================================================
# Zivid SDK 2.11 + OpenCV
# ArUco Marker Reprojection Error Validation
#
# 功能：
# 1. 离线读取 keyframe 文件夹中的 ZDF 文件
# 2. 从 ZDF 的点云中提取 RGBA 彩色图像
# 3. 使用 ArUco Detection 检测 Marker
# 4. 使用 SOLVEPNP_IPPE_SQUARE 估计 Marker 位姿
# 5. 使用 cv2.projectPoints() 计算重投影角点
# 6. 计算每个角点的误差、Mean Error、RMSE、Max Error
# 7. 保存 CSV 和重投影验证图像
#
# 按任意键切换下一张图像
# 按 Q 或 Esc 退出
#
# 不修改原来的位姿估计程序和变换矩阵文件
# ============================================================

import sys
import csv
from pathlib import Path

import yaml
import cv2
import numpy as np
import zivid


def main():

    # ========================================================
    # 1. 初始化 Zivid SDK
    # ========================================================

    # 离线读取 ZDF 文件仍然需要初始化 Zivid Application
    with zivid.Application() as app:

        # ====================================================
        # 2. 初始化 ArUco Detector
        # ====================================================

        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_250
        )

        parameters = cv2.aruco.DetectorParameters()

        detector = cv2.aruco.ArucoDetector(
            dictionary,
            parameters
        )

        # ====================================================
        # 3. 设置文件夹路径
        # ====================================================

        # 原始采集文件夹
        folder = Path(r"D:\captures_continuousMotion")

        # 筛选后的关键帧文件夹
        keyframe_folder = folder / "keyframe"

        # 读取所有关键帧 ZDF 文件
        image_files = sorted(
            keyframe_folder.glob("*.zdf")
        )

        if not image_files:
            raise FileNotFoundError(
                "No ZDF files found in keyframe folder!"
            )

        print(f"Found {len(image_files)} ZDF files.")

        # ====================================================
        # 4. 读取相机内参 intrinsics_3d.yml
        # ====================================================

        with open(
            folder / "intrinsics_3d.yml",
            "r",
            encoding="utf-8"
        ) as f:

            data = yaml.safe_load(f)

        camera_data = data["CameraIntrinsics"]

        matrix = camera_data["CameraMatrix"]
        dist = camera_data["Distortion"]

        fx = matrix["FX"]
        fy = matrix["FY"]
        cx = matrix["CX"]
        cy = matrix["CY"]

        # 相机内参矩阵
        camera_matrix = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ],
            dtype=np.float64
        )

        # 相机畸变系数
        dist_coeffs = np.array(
            [
                dist["K1"],
                dist["K2"],
                dist["P1"],
                dist["P2"],
                dist["K3"]
            ],
            dtype=np.float64
        )

        print("\nCamera Intrinsic Matrix:")
        print(camera_matrix)

        print("\nDistortion Coefficients:")
        print(dist_coeffs)

        # ====================================================
        # 5. 定义 Marker 的真实 3D 角点
        # ====================================================

        # 与原程序保持一致
        aruco_length = 0.1  # 单位：m

        half_length = aruco_length / 2.0

        # SOLVEPNP_IPPE_SQUARE 要求的角点顺序
        object_points = np.array(
            [
                [-half_length, half_length, 0.0],   # Corner 0
                [half_length, half_length, 0.0],    # Corner 1
                [half_length, -half_length, 0.0],   # Corner 2
                [-half_length, -half_length, 0.0]   # Corner 3
            ],
            dtype=np.float32
        )

        # ====================================================
        # 6. 创建验证结果保存文件夹
        # ====================================================

        output_folder = (
            keyframe_folder / "reprojection_validation"
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # 保存所有 Marker 的误差结果
        csv_results = []

        # ====================================================
        # 7. 创建图像显示窗口
        # ====================================================

        window_name = "ZDF Reprojection Error Validation"

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            window_name,
            1000,
            700
        )

        # ====================================================
        # 8. 遍历所有关键帧 ZDF 文件
        # ====================================================

        for image_path in image_files:

            print("\n===================================")
            print(f"Processing: {image_path.name}")

            # ------------------------------------------------
            # Step 1: 读取 ZDF 文件
            # ------------------------------------------------

            try:

                zivid_frame = zivid.Frame(
                    str(image_path)
                )

                # 获取点云对象
                point_cloud = zivid_frame.point_cloud()

                # 从 3D 数据中提取 RGBA 彩色图像
                rgba = point_cloud.copy_data("rgba")

                # 转换为 OpenCV BGR 格式
                frame = cv2.cvtColor(
                    rgba,
                    cv2.COLOR_RGBA2BGR
                )

            except Exception as error:

                print(
                    f"Cannot read ZDF: {image_path.name}"
                )
                print(error)
                continue

            # ------------------------------------------------
            # Step 2: 转换为灰度图像
            # ------------------------------------------------

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # ------------------------------------------------
            # Step 3: ArUco Marker Detection
            # ------------------------------------------------

            marker_corners, marker_ids, _ = (
                detector.detectMarkers(gray)
            )

            detected_count = (
                0 if marker_ids is None
                else len(marker_ids)
            )

            print(
                f"Detected markers: {detected_count}"
            )

            # =================================================
            # 9. 对每个 Marker 进行重投影误差验证
            # =================================================

            if detected_count > 0:

                # 绘制 Marker 边框和 ID
                cv2.aruco.drawDetectedMarkers(
                    frame,
                    marker_corners,
                    marker_ids
                )

                # 遍历当前图像中的每个 Marker
                for corners, marker_id in zip(
                    marker_corners,
                    marker_ids.flatten()
                ):

                    # -----------------------------------------
                    # Step 4: 提取检测到的 2D 角点
                    # -----------------------------------------

                    detected_points = corners.reshape(4, 2)

                    # -----------------------------------------
                    # Step 5: 使用 solvePnP 估计位姿
                    # -----------------------------------------

                    pose_found, rvec, tvec = cv2.solvePnP(
                        object_points,
                        detected_points,
                        camera_matrix,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )

                    if not pose_found:

                        print(
                            f"Marker {marker_id}: PnP failed"
                        )
                        continue

                    # -----------------------------------------
                    # Step 6: 输出位姿估计结果
                    # -----------------------------------------

                    print(f"\nMarker ID: {marker_id}")

                    print("rvec:")
                    print(rvec.ravel())

                    print("tvec (m):")
                    print(tvec.ravel())

                    # -----------------------------------------
                    # Step 7: 构建 Marker -> Camera 变换矩阵
                    # -----------------------------------------

                    R, _ = cv2.Rodrigues(rvec)

                    T_marker_to_camera = np.eye(
                        4,
                        dtype=np.float64
                    )

                    T_marker_to_camera[:3, :3] = R

                    T_marker_to_camera[:3, 3] = (
                        tvec.flatten()
                    )

                    print("T_marker_to_camera:")
                    print(T_marker_to_camera)

                    # 注意：
                    # 此程序只打印矩阵，不保存 .npy 文件
                    # 因此不会覆盖原来的变换矩阵

                    # -----------------------------------------
                    # Step 8: 将 3D 角点重新投影到 2D 图像
                    # -----------------------------------------

                    projected_points, _ = cv2.projectPoints(
                        object_points,
                        rvec,
                        tvec,
                        camera_matrix,
                        dist_coeffs
                    )

                    # 转换为 4×2 数组
                    projected_points = (
                        projected_points.reshape(4, 2)
                    )

                    # -----------------------------------------
                    # Step 9: 计算四个角点的重投影误差
                    # -----------------------------------------

                    # 检测角点与重投影角点的坐标差
                    differences = (
                        detected_points - projected_points
                    )

                    # 计算每个角点的欧氏距离
                    # 单位：pixel
                    corner_errors = np.linalg.norm(
                        differences,
                        axis=1
                    )

                    # -----------------------------------------
                    # Step 10: 计算整体误差
                    # -----------------------------------------

                    # 平均重投影误差
                    mean_error = np.mean(
                        corner_errors
                    )

                    # 均方根重投影误差
                    rmse = np.sqrt(
                        np.mean(corner_errors ** 2)
                    )

                    # 最大角点重投影误差
                    max_error = np.max(
                        corner_errors
                    )

                    # -----------------------------------------
                    # Step 11: 输出误差结果
                    # -----------------------------------------

                    print(
                        "Detected corner points (pixel):"
                    )
                    print(detected_points)

                    print(
                        "Reprojected corner points (pixel):"
                    )
                    print(projected_points)

                    print(
                        "Corner reprojection errors (pixel):"
                    )
                    print(corner_errors)

                    print(
                        f"Mean Error: {mean_error:.4f} px"
                    )

                    print(
                        f"RMSE:       {rmse:.4f} px"
                    )

                    print(
                        f"Max Error:  {max_error:.4f} px"
                    )

                    # -----------------------------------------
                    # Step 12: 将结果保存到 csv_results
                    # -----------------------------------------

                    csv_results.append(
                        [
                            image_path.name,
                            int(marker_id),
                            float(corner_errors[0]),
                            float(corner_errors[1]),
                            float(corner_errors[2]),
                            float(corner_errors[3]),
                            float(mean_error),
                            float(rmse),
                            float(max_error)
                        ]
                    )

                    # -----------------------------------------
                    # Step 13: 可视化角点重投影误差
                    # -----------------------------------------

                    for i in range(4):

                        # 实际检测角点
                        detected = tuple(
                            np.round(
                                detected_points[i]
                            ).astype(int)
                        )

                        # 重投影角点
                        projected = tuple(
                            np.round(
                                projected_points[i]
                            ).astype(int)
                        )

                        # 绿色圆点：实际检测角点
                        cv2.circle(
                            frame,
                            detected,
                            5,
                            (0, 255, 0),
                            -1
                        )

                        # 红色十字：重投影角点
                        cv2.drawMarker(
                            frame,
                            projected,
                            (0, 0, 255),
                            markerType=cv2.MARKER_CROSS,
                            markerSize=14,
                            thickness=2
                        )

                        # 黄色连线：两点之间的误差
                        cv2.line(
                            frame,
                            detected,
                            projected,
                            (0, 255, 255),
                            1
                        )

                        # 显示角点编号 0、1、2、3
                        cv2.putText(
                            frame,
                            str(i),
                            (
                                detected[0] + 8,
                                detected[1] - 8
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2
                        )

                    # -----------------------------------------
                    # Step 14: 在 Marker 附近显示 RMSE
                    # -----------------------------------------

                    center = np.mean(
                        detected_points,
                        axis=0
                    ).astype(int)

                    cv2.putText(
                        frame,
                        f"ID {marker_id}: {rmse:.3f} px",
                        (
                            int(center[0]),
                            int(center[1])
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        2
                    )

            # =================================================
            # 10. 显示图像信息
            # =================================================

            cv2.putText(
                frame,
                f"DICT_4X4_250 | Detected: {detected_count}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                image_path.name,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Green: Detected | Red: Reprojected",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Any key: next | Q / Esc: exit",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # =================================================
            # 11. 保存验证图像
            # =================================================

            output_image_path = (
                output_folder /
                f"{image_path.stem}_reprojection.png"
            )

            success = cv2.imwrite(
                str(output_image_path),
                frame
            )

            if success:
                print(
                    f"Validation image saved: {output_image_path}"
                )
            else:
                print(
                    f"Failed to save: {output_image_path}"
                )

            # 显示当前图像
            cv2.imshow(
                window_name,
                frame
            )

            # 等待按键
            key = cv2.waitKey(0) & 0xFF

            if (
                key in (ord("q"), ord("Q"), 27)
                or cv2.getWindowProperty(
                    window_name,
                    cv2.WND_PROP_VISIBLE
                ) < 1
            ):
                break

        # ====================================================
        # 12. 保存 CSV 误差结果
        # ====================================================

        csv_path = (
            output_folder / "reprojection_errors.csv"
        )

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.writer(f)

            # CSV 表头
            writer.writerow(
                [
                    "Image",
                    "Marker_ID",
                    "Corner_0_Error_px",
                    "Corner_1_Error_px",
                    "Corner_2_Error_px",
                    "Corner_3_Error_px",
                    "Mean_Error_px",
                    "RMSE_px",
                    "Max_Error_px"
                ]
            )

            # 写入所有 Marker 的误差数据
            writer.writerows(csv_results)

        print("\n===================================")
        print("Reprojection validation completed!")
        print(f"CSV saved to: {csv_path}")

        # ====================================================
        # 13. 输出整体统计结果
        # ====================================================

        if csv_results:

            all_rmse = np.array(
                [row[7] for row in csv_results]
            )

            print(
                f"Validated Markers: {len(all_rmse)}"
            )

            print(
                f"Average RMSE: {np.mean(all_rmse):.4f} px"
            )

            print(
                f"Maximum RMSE: {np.max(all_rmse):.4f} px"
            )

            print(
                f"Minimum RMSE: {np.min(all_rmse):.4f} px"
            )

        print("===================================")


# ============================================================
# 异常处理
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        pass

    except Exception as error:

        print(
            f"Zivid/OpenCV Validation Error: {error}",
            file=sys.stderr
        )

        sys.exit(1)

    finally:
        cv2.destroyAllWindows()