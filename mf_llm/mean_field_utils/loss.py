
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self, ring_attn_group=None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

        self.ring_attn_group = ring_attn_group
        if self.ring_attn_group:
            self.ring_attn_rank = dist.get_rank(self.ring_attn_group)
            self.ring_attn_world_size = dist.get_world_size(self.ring_attn_group)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # RingAttention
        if self.ring_attn_group is not None:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.ring_attn_world_size
            start_idx = self.ring_attn_rank * seq_len_per_process
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)
            labels = labels[..., start_idx:end_idx]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # if labels are all IGNORE_INDEX, then nn.CrossEntropyLoss will be nan
            if torch.all(shift_labels == self.IGNORE_INDEX):
                # Use mean of logits multiplied by 0 to maintain gradient flow
                loss = shift_logits.mean() * 0
            else:
                loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.ring_attn_group)
            loss = loss / self.ring_attn_world_size
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


import torch


def calculate_log_probs(model, tokenizer, prompts, true_actions, max_length, device, packing_samples=False,
                        pretrain_mode=False, ring_attn_group=None):
    """
    Calculate the Cross-Entropy Loss for a given model, prompts, and true actions.

    Args:
        model: The model to use.
        tokenizer: The tokenizer corresponding to the model.
        prompts (list of str): List of input prompts.
        true_actions (list of str): List of true action strings for each prompt.
        max_length (int): Maximum sequence length for the model.
        device (str): Device to use (e.g., "cpu" or "cuda").
        packing_samples (bool): Whether the inputs are packed samples.
        pretrain_mode (bool): Whether the model is in pretraining mode.
        ring_attn_group: Optional ring attention group for distributed training.

    Returns:
        float: Average Cross-Entropy Loss across all prompts.
    """
    # Initialize GPTLMLoss with optional ring_attn_group
    loss_fn = GPTLMLoss(ring_attn_group=ring_attn_group)
    
    processed_true_actions = [
        action if action.strip() != "" else "转发微博" for action in true_actions
    ]
    
    # Combine each prompt with its corresponding true_action
    combined_texts = [f"{prompt}{action}" for prompt, action in zip(prompts, processed_true_actions)]
    
    # Tokenize prompts
    tokenized_prompts = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    
    # Tokenize combined prompts and true actions
    tokenized_combined = tokenizer(
        combined_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    
    input_ids = tokenized_combined["input_ids"]
    labels = input_ids.clone()
    prompts_ids = tokenized_prompts["input_ids"]
    attention_mask = tokenized_combined["attention_mask"]
    
    # For each sample in the batch, set the labels corresponding to the prompt tokens to IGNORE_INDEX
    # This ensures that the loss is only computed for the true_actions
    for i in range(labels.size(0)):
        prompt_length = (prompts_ids[i] != tokenizer.pad_token_id).sum().item()
        labels[i, :prompt_length] = loss_fn.IGNORE_INDEX
    
    # Forward pass with model
    with torch.no_grad():
        if ring_attn_group is None:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_output=True
            )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_output=True,
                ring_attn_group=ring_attn_group,
            )

    logits = outputs.logits
    
    # Compute the loss
    loss = loss_fn(logits, labels)
    
    if torch.isnan(loss):
        print("Warning: Loss is NaN")
        return 10.0  # Returning a default high loss value
    
    return loss.item()

