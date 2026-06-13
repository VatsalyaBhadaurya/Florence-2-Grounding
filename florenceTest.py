from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw
import torch

MODEL_ID = "microsoft/Florence-2-large-ft"

# Reference object used for pixel -> cm scale conversion
REFERENCE_LABEL = "keyboard"
REFERENCE_WIDTH_CM = 33.1
REFERENCE_HEIGHT_CM = 14.5

# Load processor
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16
).cuda()

image = Image.open(r"D:\vatty\work\Florence-Test\image_copy.png").convert("RGB")

TASK_PROMPT = "<CAPTION_TO_PHRASE_GROUNDING>"


def ground_phrase(phrase):
    prompt = TASK_PROMPT + phrase

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].half()

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=512
    )

    result = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False
    )[0]

    parsed = processor.post_process_generation(
        result,
        task=TASK_PROMPT,
        image_size=(image.width, image.height)
    )

    return parsed[TASK_PROMPT]


# 1. Detect the reference object and compute pixel -> cm scale
ref_results = ground_phrase(REFERENCE_LABEL)
ref_bbox = ref_results["bboxes"][0]
ref_x1, ref_y1, ref_x2, ref_y2 = ref_bbox
ref_width_px = ref_x2 - ref_x1
ref_height_px = ref_y2 - ref_y1

# Note: the reference object is rotated in the image, so its axis-aligned
# bbox is larger than its true width/height -> px_per_cm (and thus all
# real-world sizes below) are approximate, not exact measurements.
px_per_cm_x = ref_width_px / REFERENCE_WIDTH_CM
px_per_cm_y = ref_height_px / REFERENCE_HEIGHT_CM
px_per_cm = (px_per_cm_x + px_per_cm_y) / 2

print(
    f"Reference '{REFERENCE_LABEL}': bbox={ref_bbox}, "
    f"{ref_width_px:.1f}px x {ref_height_px:.1f}px "
    f"-> {px_per_cm:.2f} px/cm"
)

# 2. Detect target objects
target_results = ground_phrase("mouse")
print(target_results)

# 3. Draw boxes/centers and compute real-world dimensions
draw_image = image.copy()
draw = ImageDraw.Draw(draw_image)

for i, (bbox, label) in enumerate(zip(target_results["bboxes"], target_results["labels"])):
    x1, y1, x2, y2 = bbox

    width_px = x2 - x1
    height_px = y2 - y1
    center_x = x1 + width_px / 2
    center_y = y1 + height_px / 2

    width_cm = width_px / px_per_cm
    height_cm = height_px / px_per_cm

    print(
        f"{label} #{i}: center=({center_x:.1f}, {center_y:.1f}) px, "
        f"size={width_px:.1f}x{height_px:.1f} px "
        f"-> {width_cm:.1f}cm x {height_cm:.1f}cm"
    )

    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    draw.text((x1, max(0, y1 - 20)), label, fill="red")

    r = 6
    draw.ellipse(
        [center_x - r, center_y - r, center_x + r, center_y + r],
        fill="yellow",
        outline="black"
    )

# Draw the reference box too, for sanity-checking the scale
draw.rectangle([ref_x1, ref_y1, ref_x2, ref_y2], outline="blue", width=4)
draw.text((ref_x1, max(0, ref_y1 - 20)), REFERENCE_LABEL, fill="blue")

output_path = r"D:\vatty\work\Florence-Test\output.png"
draw_image.save(output_path)
print(f"Saved visualization to {output_path}")
