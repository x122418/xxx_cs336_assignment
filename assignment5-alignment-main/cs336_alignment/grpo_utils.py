import torch
from typing import Any, Callable, Literal
from transformers import PreTrainedTokenizerBase
from torch.nn.utils.rnn import pad_sequence


def tokenize_prompt_and_output(
    prompt_strs: list[str], 
    output_strs: list[str], 
    tokenizer: PreTrainedTokenizerBase,
    ) -> dict[str, torch.Tensor]:

    assert len(prompt_strs) == len(output_strs)

    prompt_ids = tokenizer(
    prompt_strs,
    add_special_tokens=False,
    padding=False,
)["input_ids"]
    
    output_ids = tokenizer(
    output_strs,
    add_special_tokens=False,
    padding=False,
)["input_ids"]
    
    combined_ids = []
    response_mask = []
    for a, b in zip(prompt_ids, output_ids):
        combined_ids.append(torch.tensor(a+b, dtype = torch.long))
        response_mask.append(torch.tensor([False]*len(a) + [True]*len(b), dtype = torch.bool))

    combined_ids = pad_sequence(combined_ids,
                                batch_first=True,
                                padding_value=tokenizer.pad_token_id)
    
    response_mask = pad_sequence(response_mask,
                                batch_first=True,
                                padding_value=False)


    return {"input_ids":combined_ids[:,:-1],
            "labels":combined_ids[:, 1:],
            "response_mask": response_mask[:, 1:],
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:

    logits = model(input_ids)  # ..., d
    

    if return_token_entropy:
        return


    return