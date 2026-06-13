import sys
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

profile = pipeline.start(config)

try:
    device = profile.get_device()

    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale} m/unit")

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()

    color_intr = color_stream.get_intrinsics()
    depth_intr = depth_stream.get_intrinsics()

    def print_intrinsics(name, intr):
        print(f"\n{name} intrinsics:")
        print(f"  size: {intr.width} x {intr.height}")
        print(f"  fx, fy: {intr.fx:.3f}, {intr.fy:.3f}")
        print(f"  cx, cy: {intr.ppx:.3f}, {intr.ppy:.3f}")
        print(f"  distortion model: {intr.model}")
        print(f"  coeffs: {intr.coeffs}")

    print_intrinsics("Color", color_intr)
    print_intrinsics("Depth", depth_intr)

    extrinsics = depth_stream.get_extrinsics_to(color_stream)
    print("\nDepth -> Color extrinsics:")
    print(f"  rotation:    {extrinsics.rotation}")
    print(f"  translation: {extrinsics.translation}")

    # Optional: on-chip self-calibration. Requires the camera to be pointed
    # at a flat, textured surface ~0.3-1.2m away, filling most of the frame.
    if "--recalibrate" in sys.argv:
        print("\nRunning on-chip self-calibration...")
        print("Point the camera at a flat, textured surface 0.3-1.2m away.")
        input("Press Enter when ready...")

        auto_calib = device.as_auto_calibrated_device()

        def progress_callback(progress):
            print(f"  progress: {progress}%")

        table, health = auto_calib.run_on_chip_calibration("", progress_callback, 30000)
        print(f"Calibration health: {health}")

        if abs(health) < 0.25:
            auto_calib.set_calibration_table(table)
            auto_calib.write_calibration()
            print("New calibration written to device.")
        else:
            print("Health score too high (poor) - calibration NOT applied.")

finally:
    pipeline.stop()
