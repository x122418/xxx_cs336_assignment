from hw1.tokenizer_v2 import Tokenizer

vocab_path = "artifacts/tokenizers/tinystories_5M_10k/vocab_tinystories_5M_bpe_10k.pkl"
merges_path = "artifacts/tokenizers/tinystories_5M_10k/merges_tinystories_5M_bpe_10k.pkl"
special_tokens = ["<|endoftext|>"]

tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
