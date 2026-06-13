import threading
import queue

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "microsoft/Florence-2-large-ft"
GROUNDING_TASK_PROMPT = "<CAPTION_TO_PHRASE_GROUNDING>"
SEGMENTATION_TASK_PROMPT = "<REGION_TO_SEGMENTATION>"
LOC_BINS = 1000

COLOR_WIDTH, COLOR_HEIGHT, FPS = 1280, 720, 30
# D415 stereo depth is computed natively at 848x480; requesting 1280x720
# depth forces firmware upscaling and noticeably degrades accuracy.
DEPTH_WIDTH, DEPTH_HEIGHT = 848, 480
# Shrink both views before displaying side by side so the combined window fits on screen
DISPLAY_SCALE = 0.5

print("Loading Florence-2 model...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16
).cuda()


def run_florence(prompt, image, task_prompt, max_new_tokens=512):
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].half()

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens
    )
    result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    return processor.post_process_generation(
        result, task=task_prompt, image_size=(image.width, image.height)
    )[task_prompt]


def ground_phrase(image, phrase):
    return run_florence(GROUNDING_TASK_PROMPT + phrase, image, GROUNDING_TASK_PROMPT)


def bbox_to_loc_tokens(bbox, width, height):
    x1, y1, x2, y2 = bbox
    qx1 = min(int(x1 * LOC_BINS / width), LOC_BINS - 1)
    qy1 = min(int(y1 * LOC_BINS / height), LOC_BINS - 1)
    qx2 = min(int(x2 * LOC_BINS / width), LOC_BINS - 1)
    qy2 = min(int(y2 * LOC_BINS / height), LOC_BINS - 1)
    return f"<loc_{qx1}><loc_{qy1}><loc_{qx2}><loc_{qy2}>"


def segment_region(image, bbox):
    """Return the polygon parts (list of flat [x,y,...] coordinate lists) for the
    object inside bbox, or None if segmentation returned nothing usable."""
    loc_tokens = bbox_to_loc_tokens(bbox, image.width, image.height)
    prompt = SEGMENTATION_TASK_PROMPT + loc_tokens

    result = run_florence(prompt, image, SEGMENTATION_TASK_PROMPT, max_new_tokens=1024)
    polygons = result.get("polygons")
    if not polygons or not polygons[0]:
        return None
    return polygons[0]


def median_depth_m(depth_image, depth_scale, mask):
    region = depth_image[mask > 0]
    valid = region[region > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * depth_scale


# Background thread: read object queries from the console without blocking the video loop
query_queue = queue.Queue()


def input_worker():
    while True:
        text = input()
        query_queue.put(text.strip())


threading.Thread(target=input_worker, daemon=True).start()

# RealSense pipeline setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# Depth post-processing chain (applied in disparity domain for spatial/temporal
# filtering, then hole-filling) to reduce noise and fill invalid depth pixels.
depth_to_disparity = rs.disparity_transform(True)
spatial_filter = rs.spatial_filter()
temporal_filter = rs.temporal_filter()
disparity_to_depth = rs.disparity_transform(False)
hole_filling_filter = rs.hole_filling_filter()


def filter_depth_frame(depth_frame):
    f = depth_to_disparity.process(depth_frame)
    f = spatial_filter.process(f)
    f = temporal_filter.process(f)
    f = disparity_to_depth.process(f)
    f = hole_filling_filter.process(f)
    return f.as_depth_frame()


depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
# fx and fy are nearly identical on the D415; a single average focal length
# is used since the oriented box below can be at any in-plane angle.
avg_focal_px = (intrinsics.fx + intrinsics.fy) / 2

print("Camera ready. Type an object name (e.g. 'tomato') and press Enter to locate it.")
print("Type 'quit' to exit.")

current_label = None
current_polygon_parts = None
current_box_points = None
current_center = None
current_dims_cm = None
current_distance_m = None

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        depth_frame = filter_depth_frame(depth_frame)

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        h, w = depth_image.shape

        # Handle a pending text query
        if not query_queue.empty():
            text = query_queue.get()
            if text.lower() in ("quit", "exit"):
                break

            if text:
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                results = ground_phrase(pil_image, text)

                if results["bboxes"]:
                    bbox = results["bboxes"][0]
                    label = results["labels"][0]
                    x1, y1, x2, y2 = bbox

                    polygon_parts = segment_region(pil_image, bbox)

                    # Build a mask for depth sampling and a point set for the
                    # oriented bounding box, from the segmentation mask if
                    # available, otherwise fall back to the axis-aligned bbox.
                    mask = np.zeros((h, w), dtype=np.uint8)
                    if polygon_parts:
                        all_points = []
                        for part in polygon_parts:
                            pts = np.array(part, dtype=np.float32).reshape(-1, 2)
                            all_points.append(pts)
                            cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
                        all_points = np.vstack(all_points)
                    else:
                        all_points = np.array(
                            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                        )
                        cv2.fillPoly(mask, [all_points.astype(np.int32)], 255)

                    rect = cv2.minAreaRect(all_points)
                    (rect_cx, rect_cy), (rect_w, rect_h), _angle = rect
                    box_points = cv2.boxPoints(rect)

                    z = median_depth_m(depth_image, depth_scale, mask)
                    if z is not None and z > 0:
                        width_cm = (rect_w * z * 100) / avg_focal_px
                        height_cm = (rect_h * z * 100) / avg_focal_px
                        current_dims_cm = (width_cm, height_cm)
                        current_distance_m = z
                    else:
                        current_dims_cm = None
                        current_distance_m = None

                    current_label = label
                    current_polygon_parts = polygon_parts
                    current_box_points = box_points
                    current_center = (rect_cx, rect_cy)

                    if current_dims_cm is not None:
                        w_cm, h_cm = current_dims_cm
                        print(
                            f"{label}: {w_cm:.1f}cm x {h_cm:.1f}cm, "
                            f"dist: {current_distance_m * 100:.1f}cm"
                        )
                    else:
                        print(f"{label}: found, but depth unavailable")
                else:
                    print(f"'{text}' not found in frame.")
                    current_label = None
                    current_polygon_parts = None
                    current_box_points = None
                    current_center = None

        # Draw the last detection on the current frame
        if current_box_points is not None:
            box_pts_int = np.int32(current_box_points)
            cx, cy = int(current_center[0]), int(current_center[1])

            cv2.drawContours(color_image, [box_pts_int], 0, (0, 0, 255), 2)
            if current_polygon_parts:
                for part in current_polygon_parts:
                    pts = np.array(part, dtype=np.int32).reshape(-1, 2)
                    cv2.polylines(color_image, [pts], True, (0, 255, 0), 2)
            cv2.circle(color_image, (cx, cy), 5, (0, 255, 255), -1)

            info_lines = [current_label]
            if current_dims_cm is not None:
                w_cm, h_cm = current_dims_cm
                info_lines.append(f"{w_cm:.1f}cm x {h_cm:.1f}cm")
            if current_distance_m is not None:
                info_lines.append(f"dist: {current_distance_m * 100:.1f}cm")

            text_x = int(np.min(box_pts_int[:, 0]))
            text_y = int(np.min(box_pts_int[:, 1]))
            for i, line in enumerate(info_lines):
                cv2.putText(
                    color_image, line,
                    (text_x, max(15, text_y - 10 - 20 * (len(info_lines) - 1 - i))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        if current_box_points is not None:
            cv2.drawContours(depth_colormap, [box_pts_int], 0, (0, 0, 255), 2)

        display_color = cv2.resize(color_image, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        display_depth = cv2.resize(depth_colormap, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

        combined = np.hstack((display_color, display_depth))
        cv2.imshow("Florence-2 + RealSense (color | depth)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
