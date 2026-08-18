from hw1.tokenizer_v2 import Tokenizer

from pathlib import Path
import yaml, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

def passage_iter(f, seperation_token):
    buffer = []
    for row in f:
        buffer.append(row)
        if seperation_token in row:
            passage = "".join(buffer)
            buffer =  []
            yield passage            

    if buffer:
        passage = "".join(buffer)
        yield passage
        

def main():
    # 读取超参数配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    vocab_path = config['vocab_path']
    merges_path = config['merges_path']
    special_tokens = config["special_tokens"]

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    input_path = config['data']['input_path']
    output_path = config['data']['output_path']

    with open(input_path, "r", encoding="utf-8") as f:
        passages = passage_iter(f, special_tokens[0])
        encoded_id = tokenizer.encode_iterable(passages)

        token_array = np.fromiter(
            encoded_id,
            dtype=np.uint16,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        np.save(output_path, token_array)

if __name__ == "__main__":
    main()