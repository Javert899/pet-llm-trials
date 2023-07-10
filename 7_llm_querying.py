from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pretrained model and tokenizer
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Encode input context
while True:
    input_context = input("\n\n\nInsert a query -> ")
    input_ids = tokenizer.encode(input_context, return_tensors='pt')

    # Generate the output
    output = model.generate(input_ids)

    # Decode the output
    for text in output:
        generated_text = tokenizer.decode(text, skip_special_tokens=True)
        print(generated_text)
