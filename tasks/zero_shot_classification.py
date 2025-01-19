from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning() # Hide warning messages about torchvision
import os
os.environ["HF_HOME"] = "/home/user/shared_read_only/huggingface_cache/" # Remove this line if running any custom model or if loading the model fails
from transformers import pipeline

pipe = pipeline("zero-shot-classification")
print(f"The pipeline is using the model: {pipe.model.config.name_or_path}")
print(f"Execution device: {'GPU' if pipe.device.type == 'cuda' else 'CPU'}\n")

text = "I love studying data science and AI."
labels = ["technology", "sports", "politics"]
result = pipe(text, candidate_labels=labels)

print(f"Text: {text}")
print(f"Labels: {labels}")
print(f"Predicted Label: {result['labels'][0]}, Score: {result['scores'][0]:.4f}")
print("Full Scores:")
for label, score in zip(result["labels"], result["scores"]):
    print(f"  {label}: {score:.4f}")