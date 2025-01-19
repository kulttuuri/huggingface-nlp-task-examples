from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning() # Hide warning messages about torchvision
import os
os.environ["HF_HOME"] = "/home/user/shared_read_only/huggingface_cache/" # Remove this line if running any custom model or if loading the model fails
from transformers import pipeline

pipe = pipeline("question-answering")
print(f"The pipeline is using the model: {pipe.model.config.name_or_path}")
print(f"Execution device: {'GPU' if pipe.device.type == 'cuda' else 'CPU'}\n")

context = "The capital of France is Paris. It is known for its art, fashion, and culture."
question = "What is the capital of France?"
result = pipe(question=question, context=context)

print(f"Question: {question}\nAnswer: {result['answer']}, Score: {result['score']:.4f}\n")