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
    from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
    from transformers import DPRQuestionEncoder, DPRContextEncoder
    import torch

    # Initialize the question encoder and context encoder
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

    # Initialize the tokenizer
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")

    # Initialize the retriever
    retriever = RagRetriever(
        question_encoder=question_encoder,
        context_encoder=context_encoder,
        index_name="exact",  # Use the exact index for faster retrieval
        use_dummy_dataset=True  # Use a dummy dataset
    )

    # Initialize the generator
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

    # Define a question and a context
    context = "The Eiffel Tower is located in Paris."
    question = "Where is the Eiffel Tower located?"

    # Encode the question and context
    inputs = tokenizer(question, context, return_tensors="pt")

    # Generate the output
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        decoder_start_token_id=model.config.pad_token_id
    )

    # Decode the output
    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Print the answer
    print(generated[0])
