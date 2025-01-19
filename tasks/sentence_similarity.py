from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()  # Hide warning messages about torchvision
import os
#os.environ["HF_HOME"] = "/home/user/shared_read_only/huggingface_cache/"
from transformers import pipeline

pipe = pipeline("text-classification", model="sentence-transformers/all-MiniLM-L6-v2")
print(f"The pipeline is using the model: {pipe.model.config.name_or_path}")
print(f"Execution device: {'GPU' if pipe.device.type == 'cuda' else 'CPU'}\n")

sentence_1 = "I love studying data science and AI."
sentence_2 = "Artificial Intelligence and data science are my passion."
text = f"{sentence_1} [SEP] {sentence_2}"

result = pipe(text)

print(f"Sentence 1: {sentence_1}")
print(f"Sentence 2: {sentence_2}")
print(f"Similarity Score: {result[0]['score']:.4f}")