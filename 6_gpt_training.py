from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def fine_tune_gpt2(model_name, train_path, output_dir, eval_path=None, epochs=4, learning_rate=5e-5):
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Load training dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)

    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
    )

    # Load validation dataset if provided
    eval_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=eval_path,
        block_size=128) if eval_path else None

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=2,
        learning_rate=learning_rate,
        evaluation_strategy="epoch" if eval_path else "no",
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train model
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)

# Fine-tune GPT-2 on your text data
fine_tune_gpt2("gpt2", "./pet_united.txt", "./gpt2_finetuned", "./pet_united.txt", epochs=100, learning_rate=1e-5)

