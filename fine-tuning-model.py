# Fine Tuning Model to Classify Disease based on Symptoms
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import json
import requests

# Step 1: Load Dataset
dataset = load_dataset("duxprajapati/symptom-disease-dataset")

# Step 2: Load Mapping and Process Labels
url = "https://huggingface.co/datasets/duxprajapati/symptom-disease-dataset/resolve/main/mapping.json"
response = requests.get(url)
label_mapping = json.loads(response.text)
reverse_mapping = {v: k for k, v in label_mapping.items()}

# Map numeric labels to strings
def map_numeric_to_string(example):
    example["label"] = reverse_mapping[example["label"]]
    return example

dataset = dataset.map(map_numeric_to_string)

# Step 3: Tokenize Dataset
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Step 4: Encode Labels
labels = list(label_mapping.keys())  # List of all disease names
label_encoder = LabelEncoder()
label_encoder.fit(labels)

def encode_labels(example):
    example["label"] = label_encoder.transform([example["label"]])[0]
    return example

tokenized_datasets = tokenized_datasets.map(encode_labels)

# Step 5: Load Model
num_labels = len(labels)
model = AutoModelForSequenceClassification.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT", num_labels=num_labels
)

# Step 6: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Reduce batch size for memory constraints
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    fp16=False,  
)

# Step 7: Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

# Step 8: Train the Model
trainer.train()

# Step 9: Save the Model
trainer.save_model("./fine_tuned_clinicalbert")
tokenizer.save_pretrained("./fine_tuned_clinicalbert")
