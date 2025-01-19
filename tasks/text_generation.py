from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning() # Hide warning messages about torchvision
import os
os.environ["HF_HOME"] = "/home/user/shared_read_only/huggingface_cache/" # Remove this line if running any custom model or if loading the model fails
from transformers import pipeline

pipe = pipeline("text-generation")
print(f"The pipeline is using the model: {pipe.model.config.name_or_path}")

prompt = "Once upon a time in a small village"
result = pipe(prompt, max_length=50, num_return_sequences=1)

print(f"Prompt: {prompt}\nGenerated Text: {result[0]['generated_text']}")