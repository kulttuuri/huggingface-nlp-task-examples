from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning() # Hide warning messages about torchvision
import os
os.environ["HF_HOME"] = "/home/user/shared_read_only/huggingface_cache/" # Remove this line if running any custom model or if loading the model fails
from transformers import pipeline

pipe = pipeline("text-classification")
print(f"The pipeline is using the model: {pipe.model.config.name_or_path}")
print(f"Execution device: {'GPU' if pipe.device.type == 'cuda' else 'CPU'}\n")

prompts = ["This restaurant is awesome", "I hate this restaurant"]

result = pipe(prompts)

# Extract and print the result along with the original prompt
for prompt, res in zip(prompts, result):
    label = res['label']
    score = res['score']
    print(f"Prompt: {prompt}\nLabel: {label}, Score: {score:.4f}\n")