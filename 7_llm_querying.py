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
    from transformers import BartForQuestionAnswering, BartTokenizer

    # Load the pre-trained BART model and tokenizer

    # Provide the context and the question
    context = "OpenAI is an artificial intelligence research lab consisting of the for-profit OpenAI LP and its parent company, the non-profit OpenAI Inc. OpenAI LP is an employer of researchers and engineers who aim to ensure that artificial general intelligence (AGI) benefits all of humanity."
    while True:
        question = input("Please insert a question -> ")

        model = BartForQuestionAnswering.from_pretrained('facebook/bart-large')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

        # Encode the context and question to get input_ids and attention_mask
        inputs = tokenizer(question, context, return_tensors='pt')

        # Feed the input to BART to retrieve the start and end positions of the answer
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])
        outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)

        # Get the loss and the start/end position logits
        loss = outputs.loss
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # Find the tokens with the highest `start` and `end` scores.
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores) + 1

        # Get the answer from the context
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

        print(f"The answer to the question is: {answer}")
