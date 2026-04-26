import torch

def calculate_text_metrics(model, tokenizer, text):
    """
    Calculates the Total NLL, per-token loss, and token count for a given text.
    
    Args:
        model: An initialized Hugging Face CausalLM.
        tokenizer: The corresponding initialized tokenizer.
        text (str): The text to evaluate.
        
    Returns:
        total_nll (float): The sum of the negative log probabilities.
        loss_per_token (list[float]): The negative log probability of each predicted token.
        amount_of_tokens (int): The total number of tokens in the input text.
    """
    # 1. Tokenize and move to the model's device
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    
    amount_of_tokens = input_ids.size(1)
    
    # 2. Forward pass to get logits (no gradients needed)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # Shape: (batch_size=1, seq_len, vocab_size)
        
    # 3. Shift logits and labels
    # We use logits up to the second-to-last token to predict labels from the second token onward.
    shift_logits = logits[0, :-1, :].contiguous()
    shift_labels = input_ids[0, 1:].contiguous()
    
    # 4. Calculate unreduced loss (loss per token)
    # reduction="none" ensures we get an array of losses rather than a single mean value.
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss_per_token_tensor = loss_fct(shift_logits, shift_labels)
    
    # Convert the tensor to a standard Python list of floats
    loss_per_token = loss_per_token_tensor.tolist()
    
    # Total NLL is simply the sum of all individual token losses
    total_nll = sum(loss_per_token)
    
    return total_nll, loss_per_token, amount_of_tokens