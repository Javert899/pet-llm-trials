import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, GPT2Config
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text_file, block_size=1024):
        self.text_file = text_file
        self.text = []
        self.block_size = block_size
        with open(text_file, "r", encoding="utf-8") as file:
            for line in file.readlines():
                self.text.append(line)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.input_ids = self.tokenizer('\n'.join(self.text), return_tensors='pt').input_ids[0]
        self.input_ids = self._chunk_text(self.input_ids)

    def _chunk_text(self, input_ids):
        ids = [input_ids[i:i+self.block_size] for i in range(0, len(input_ids), self.block_size)]
        if len(ids[-1]) < self.block_size:
            ids[-1] = torch.cat([ids[-1], torch.tensor([self.tokenizer.eos_token_id]*(self.block_size-len(ids[-1])))])
        return ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]


# Create a new tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Create a new model with a custom number of tokens
# "max_new_tokens" is the new maximum number of tokens, you might need to adjust this number
max_new_tokens = 100000
config = GPT2Config(vocab_size=tokenizer.vocab_size + max_new_tokens)
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
