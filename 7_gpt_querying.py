from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pretrained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#model = GPT2LMHeadModel.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("./gpt2_finetuned")

# Encode input context
while True:
    input_context = input("\n\n\nInsert a query -> ")
    input_ids = tokenizer.encode(input_context, return_tensors='pt')

    # Generate text
    output = model.generate(input_ids, max_length=100, temperature=0.7, num_return_sequences=1)

    # Decode the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True).split(input_context)[-1]

    print(generated_text)

