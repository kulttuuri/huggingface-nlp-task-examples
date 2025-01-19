from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning() # Hide warning messages about torchvision
import os
os.environ["HF_HOME"] = "/home/user/shared_read_only/huggingface_cache/" # Remove this line if running any custom model or if loading the model fails
from transformers import pipeline

pipe = pipeline("token-classification")
print(f"The pipeline is using the model: {pipe.model.config.name_or_path}")
print(f"Execution device: {'GPU' if pipe.device.type == 'cuda' else 'CPU'}\n")

prompt = "John lives in New York and works at Google. Anna lives in Hamburg."
result = pipe(prompt)

print(f"Prompt: {prompt}")
for res in result:
    print(f"Token: {res['word']}, Entity: {res['entity']}, Score: {res['score']:.4f}")