from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pretrained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("./gpt2_finetuned")
#tokenizer = T5Tokenizer.from_pretrained('t5-base')
#model = T5ForConditionalGeneration.from_pretrained('t5-base')
#tokenizer = T5Tokenizer.from_pretrained('t5-large')
#model = T5ForConditionalGeneration.from_pretrained('t5-large')

# Encode input context
while True:
    input_context = input("\n\n\nInsert a query -> ")
    print(input_context)
    input_ids = tokenizer.encode(input_context, return_tensors='pt')

    # Generate the output
    output = model.generate(input_ids, max_length=200)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    print(generated_text)
