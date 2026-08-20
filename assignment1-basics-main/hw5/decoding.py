from hw1.tokenizer import Tokenizer
from hw2.transformer import Transformer_LM
import torch

def decode_from_llm(
    model: Transformer_LM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    context_length: int, 
    temperature=1.0,
    top_p=1,
):
    model.eval()
    with torch.inference_mode():
        device = next(model.parameters()).device
        prompt_tokens = tokenizer.encode(prompt)
        prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
        generated_tokens = prompt_tokens
        eos_id = tokenizer.anti_vocab[
            tokenizer.special_tokens[0].encode('utf-8')
            ]

        for _ in range(max_new_tokens):
            model_input = generated_tokens[:, -context_length:]
            logits = model(model_input) # [1, S, V]
            last_position_logits = logits[:, -1, :]
            temp_last_logits = last_position_logits / temperature
            probs = torch.softmax(temp_last_logits, dim = -1)

            sorted_probs, sorted_indices = torch.sort(
                probs, dim = -1, descending=True
            )
            cumulative_probs = torch.cumsum(sorted_probs, dim = -1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            mask = torch.zeros_like(sorted_mask).scatter(
                -1, sorted_indices, sorted_mask
            )
            probs = probs.masked_fill(mask, 0)
            probs = probs / probs.sum(dim = -1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_tokens = torch.cat([generated_tokens, next_token], dim = -1)
            if next_token.item() == eos_id:
                break
        token_ids = generated_tokens[0].tolist()
    return tokenizer.decode(token_ids)
