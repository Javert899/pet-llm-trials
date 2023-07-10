from transformers import T5ForConditionalGeneration, T5Tokenizer, TextDataset, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments


def fine_tune_t5(model_name, train_path, output_dir, eval_path=None, epochs=4, learning_rate=5e-5):
    # Load model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Load training dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)

    # Define data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
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


# Fine-tune T5 on your text data
fine_tune_t5("t5-base", "./pet_united.txt", "./t5_finetuned", "/path/to/your/eval_textfile.txt",
             epochs=4, learning_rate=5e-5)
