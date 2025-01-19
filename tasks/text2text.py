from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()  # Hide warning messages about torchvision
import os
os.environ["HF_HOME"] = "/home/user/shared_read_only/huggingface_cache/" # Remove this line if running any custom model or if loading the model fails
from transformers import pipeline

pipe = pipeline("text2text-generation")
print(f"The pipeline is using the model: {pipe.model.config.name_or_path}")
print(f"Execution device: {'GPU' if pipe.device.type == 'cuda' else 'CPU'}\n")

text = "Translate the following English text to French: HuggingFace provides great NLP models."
result = pipe(text)

print(f"Input: {text}")
print(f"Generated Output: {result[0]['generated_text']}")