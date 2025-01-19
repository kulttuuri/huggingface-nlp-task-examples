from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning() # Hide warning messages about torchvision
import os
os.environ["HF_HOME"] = "/home/user/shared_read_only/huggingface_cache/" # Remove this line if running any custom model or if loading the model fails
from transformers import pipeline

pipe = pipeline("table-question-answering")
print(f"The pipeline is using the model: {pipe.model.config.name_or_path}")
print(f"Execution device: {'GPU' if pipe.device.type == 'cuda' else 'CPU'}\n")

table = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": ["25", "30", "35"],
    "City": ["New York", "San Francisco", "Los Angeles"]
}
question = "Who is 30 years old?"
result = pipe(table=table, query=question)

print(f"Question: {question}\nAnswer: {result['answer']}")