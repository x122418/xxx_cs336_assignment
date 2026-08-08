import os
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def read_file_in_chunks(input_path, chunk_size = 1024*1024):
    with open(input_path, 'r', encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    vocab = {num:bytes([num]) for num in range(256)}
    merges = []

    for idx, special_token in enumerate(special_tokens):
        vocab[idx+256] = special_token.encode("utf-8")

    with open(input_path, 'r', encoding="utf-8") as f:
        document = f.read()
    
    special_token_pattern = '|'.join(re.escape(tok) for tok in special_tokens)
    if special_tokens != []:
        document_sep_list = re.split(special_token_pattern, document)
    else:
        document_sep_list = [document]

    for segment in document_sep_list:
        

    return vocab, merges