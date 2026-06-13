from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

MODEL_ID = "microsoft/Florence-2-large-ft"

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

image = Image.open(r"D:\vatty\work\Florence-Test\image.png").convert("RGB")

# Florence-2 task token for grounding a specific phrase to a bounding box
task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
text_input = "tomatoes"
prompt = task_prompt + text_input

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
    task=task_prompt,
    image_size=(image.width, image.height)
)

print(parsed)