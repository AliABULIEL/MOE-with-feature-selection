import torch
from datasets import load_dataset
from typing import Dict, List


def calculate_text_metrics(model, tokenizer, text, max_length=None):
    """
    Calculates the Total NLL, per-token loss, and token count for a given text.

    Args:
        model: An initialized Hugging Face CausalLM.
        tokenizer: The corresponding initialized tokenizer.
        text (str): The text to evaluate.
        max_length (int, optional): The maximum sequence length.

    Returns:
        total_nll (float): The sum of the negative log probabilities.
        loss_per_token (list[float]): The negative log probability of each predicted token.
        amount_of_tokens (int): The total number of tokens in the input text.
    """
    # 1. Tokenize and move to the model's device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=max_length is not None
    ).to(model.device)
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


def load_dataset_samples(dataset_name: str, max_samples: int) -> List[Dict]:

    # Handle case where config dictionary is passed instead of max_samples integer (e.g. in some HC pipelines)
    if isinstance(max_samples, dict):
        config_dict = max_samples
        if "max_samples" in config_dict:
            max_samples_field = config_dict["max_samples"]
            if isinstance(max_samples_field, dict):
                max_samples = max_samples_field.get(dataset_name, 1000)
            elif isinstance(max_samples_field, int):
                max_samples = max_samples_field
            else:
                max_samples = 1000
        else:
            max_samples = 1000

    print(f"Loading {dataset_name} dataset (max {max_samples} samples)...")

    if dataset_name == "lambada":
        try:
            dataset = load_dataset("lambada", split="test")
        except:
            dataset = load_dataset("EleutherAI/lambada_openai", "en", split="test")
        samples = [
            {"text": item.get("text", "")}
            for item in dataset
            if item.get("text", "").strip()
        ]
    elif dataset_name == "hellaswag":
        dataset = load_dataset("hellaswag", split="validation")
        samples = []
        for item in dataset:
            ctx, endings, label = (
                item.get("ctx", ""),
                item.get("endings", []),
                int(item.get("label", 0)),
            )
            if ctx and endings and 0 <= label < len(endings):
                samples.append({"text": f"{ctx} {endings[label]}"})
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        samples = [
            {"text": item.get("text", "")}
            for item in dataset
            if item.get("text", "").strip() and len(item.get("text", "").split()) > 10
        ]
    elif dataset_name == "tinystories":
        dataset = load_dataset("roneneldan/TinyStories", split="validation")
        samples = [
            {"text": item.get("text", "")}
            for item in dataset
            if item.get("text", "").strip()
        ]
    elif dataset_name in ["fineweb-edu", "HuggingFaceFW/fineweb-edu"]:
        try:
            dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        except Exception:
            try:
                dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="default", split="train", streaming=True)
            except Exception:
                dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

        samples = []
        for item in dataset:
            text = item.get("text", "")
            language = item.get("language", "")
            if language == "en" and text.strip():
                samples.append({"text": text})
                if len(samples) >= max_samples:
                    break
    elif dataset_name in ["race", "ehovy/race"]:
        try:
            dataset = load_dataset("ehovy/race", name="all", split="test", streaming=True)
        except Exception:
            try:
                dataset = load_dataset("ehovy/race", name="all", split="validation", streaming=True)
            except Exception:
                try:
                    dataset = load_dataset("ehovy/race", name="high", split="test", streaming=True)
                except Exception:
                    dataset = load_dataset("ehovy/race", name="all", split="train", streaming=True)

        samples = []
        seen_ids = set()
        for item in dataset:
            ex_id = item.get("example_id")
            article = item.get("article", "")
            if ex_id not in seen_ids and article.strip():
                seen_ids.add(ex_id)
                samples.append({"text": article})
                if len(samples) >= max_samples:
                    break
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    samples = samples[:max_samples]
    print(f"✅ Loaded {len(samples)} samples from {dataset_name}")
    return samples
