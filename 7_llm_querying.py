from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GPT2LMHeadModel, GPT2Tokenizer


if False:
    # Load pretrained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
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
else:
    import torch
    from transformers import BertForQuestionAnswering, BertTokenizer

    # Load pretrained model and tokenizer
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # Provide your context and question
    context = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge."
    question = "Where is Hugging Face Inc. based?"

    # Encode the context and question
    input_ids = tokenizer.encode(question, context)

    # Find the tokens for the start and end of answer
    answer_start = input_ids.index(tokenizer.sep_token_id) + 1
    answer_end = len(input_ids) - 1

    # Define segment_ids: 0 for the question and 1 for the context
    segment_ids = [0] * answer_start + [1] * (answer_end - answer_start + 1)

    # Convert to tensors and run through the model
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

    # Find the tokens with the highest start and end scores
    #answer_start = torch.argmax(start_scores)
    #answer_end = torch.argmax(end_scores)

    # Convert tokens to string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))

    print(answer)