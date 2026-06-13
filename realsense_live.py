import threading
import queue

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "microsoft/Florence-2-large-ft"
TASK_PROMPT = "<CAPTION_TO_PHRASE_GROUNDING>"

COLOR_WIDTH, COLOR_HEIGHT, FPS = 1280, 720, 30
# D415 stereo depth is computed natively at 848x480; requesting 1280x720
# depth forces firmware upscaling and noticeably degrades accuracy.
DEPTH_WIDTH, DEPTH_HEIGHT = 848, 480

print("Loading Florence-2 model...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16
).cuda()


def ground_phrase(image, phrase):
    prompt = TASK_PROMPT + phrase

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].half()

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=512
    )
    result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed = processor.post_process_generation(
        result, task=TASK_PROMPT, image_size=(image.width, image.height)
    )
    return parsed[TASK_PROMPT]


def median_depth_m(depth_image, depth_scale, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = depth_image.shape

    # Sample the center 50% of the bbox to avoid edge/background noise
    cx1 = x1 + (x2 - x1) // 4
    cx2 = x2 - (x2 - x1) // 4
    cy1 = y1 + (y2 - y1) // 4
    cy2 = y2 - (y2 - y1) // 4

    cx1, cy1 = max(0, cx1), max(0, cy1)
    cx2, cy2 = min(w, cx2), min(h, cy2)

    region = depth_image[cy1:cy2, cx1:cx2]
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
fx, fy = intrinsics.fx, intrinsics.fy

print("Camera ready. Type an object name (e.g. 'tomato') and press Enter to locate it.")
print("Type 'quit' to exit.")

current_label = None
current_bbox = None
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
                    width_px = x2 - x1
                    height_px = y2 - y1

                    z = median_depth_m(depth_image, depth_scale, bbox)
                    if z is not None and z > 0:
                        width_cm = (width_px * z * 100) / fx
                        height_cm = (height_px * z * 100) / fy
                        current_dims_cm = (width_cm, height_cm)
                        current_distance_m = z
                    else:
                        current_dims_cm = None
                        current_distance_m = None

                    current_label = label
                    current_bbox = bbox
                else:
                    print(f"'{text}' not found in frame.")
                    current_label = None
                    current_bbox = None

        # Draw the last detection on the current frame
        if current_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in current_bbox]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(color_image, (cx, cy), 5, (0, 255, 255), -1)

            info_lines = [current_label]
            if current_dims_cm is not None:
                w_cm, h_cm = current_dims_cm
                info_lines.append(f"{w_cm:.1f}cm x {h_cm:.1f}cm")
            if current_distance_m is not None:
                info_lines.append(f"dist: {current_distance_m * 100:.1f}cm")

            for i, line in enumerate(info_lines):
                cv2.putText(
                    color_image, line, (x1, max(15, y1 - 10 - 20 * (len(info_lines) - 1 - i))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        if current_bbox is not None:
            cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (0, 0, 255), 2)

        combined = np.hstack((color_image, depth_colormap))
        cv2.imshow("Florence-2 + RealSense (color | depth)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
