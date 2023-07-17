import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, GPT2Config
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text_file):
        self.text_file = text_file
        self.text = []
        with open(text_file, "r", encoding="utf-8") as file:
            for line in file.readlines():
                self.text.append(line)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.encodings = self.tokenizer('\n'.join(self.text), return_tensors='pt')

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.encodings.input_ids[0]

# Create a new tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Create a new model with a custom number of tokens
# "max_new_tokens" is the new maximum number of tokens, you might need to adjust this number
max_new_tokens = 100000
config = GPT2LMHeadModel(config=GPT2Config(vocab_size=tokenizer.vocab_size + max_new_tokens))
model = GPT2LMHeadModel(config)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    weight_decay=0.01,
)

# Prepare the datasets
# "folder" is the directory where your txt files are
folder = "txt"
text_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.txt')]
datasets = [TextDataset(os.path.join(folder, text_file)) for text_file in text_files]
dataset = torch.utils.data.ConcatDataset(datasets)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

trainer.save_model("./trained_llm")
