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

prompt = "find where is tomatoes and where exactly is the tomatoes in the image, and give me the coordinates of the tomatoes in the image, and also give me the confidence score of the tomatoes in the image, and also give me the size of the tomatoes in the image, and also give me the color of the tomatoes in the image, and also give me the shape of the tomatoes in the image, and also give me the texture of the tomatoes in the image, and also give me the smell of the tomatoes in the image, and also give me the taste of the tomatoes in the image, and also give me the nutritional value of the tomatoes in the image, and also give me the health benefits of the tomatoes in the image, and also give me the recipes that can be made with tomatoes in the image, and also give me the history of tomatoes in the image, and also give me any other information about tomatoes in the image."

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

print(result)