from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning() # Hide warning messages about torchvision
import os
os.environ["HF_HOME"] = "/home/user/shared_read_only/huggingface_cache/" # Remove this line if running any custom model or if loading the model fails
from transformers import pipeline

pipe = pipeline("summarization", min_length = 30, max_length = 60)
print(f"The pipeline is using the model: {pipe.model.config.name_or_path}")

text = "HuggingFace is a leader in natural language processing technologies, " \
    "offering an extensive collection of pre-trained models through the Transformers library. " \
    "These models cover a wide range of tasks, including text summarization, question answering, " \
    "and text generation. With user-friendly APIs and a thriving open-source community, " \
    "HuggingFace has made cutting-edge NLP accessible to researchers and developers worldwide. " \
    "Their tools are widely adopted in both academia and industry, helping accelerate innovation in AI."
result = pipe(text)

print(f"\nOriginal Text: {text}\n\nSummary: {result[0]['summary_text']}")