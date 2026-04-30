from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from pathlib import Path

checkpoint_path = Path("models/model4_nlp_classification/saved_model/checkpoint-57060")
final_path = Path("models/model4_nlp_classification/saved_model/final_model")

model = DistilBertForSequenceClassification.from_pretrained(
    checkpoint_path,
    local_files_only=True
)

tokenizer = DistilBertTokenizerFast.from_pretrained(
    checkpoint_path,
    local_files_only=True
)

model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

print("Final model saved successfully.")