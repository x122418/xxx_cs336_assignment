from hw1.train_bpe import train_bpe

from pathlib import Path
import yaml, argparse
import time
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()


def main():
    # 读取超参数配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_name = config['experiment']['name']
    tokenizer_config = config["tokenizer"]
    data_cfg = config["data"]
    special_tokens = tokenizer_config["special_tokens"]
    vocab_size = tokenizer_config["vocab_size"]
    output_cfg = config["output"]

    tokenizer_dir = Path(output_cfg["directory"])
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    training_start_time = time.perf_counter()

    vocab, merges = train_bpe(data_cfg["input_path"], vocab_size, special_tokens)
    print(f"spent time{time.perf_counter() - training_start_time :.4f} s")

    vocab_path = tokenizer_dir / f"vocab_{data_name}.pkl"
    merges_path = tokenizer_dir / f"merges_{data_name}.pkl"

    with open(vocab_path, "wb") as vocab_f:
        pickle.dump(vocab, vocab_f)
    with open(merges_path, "wb") as merges_f:
            pickle.dump(merges, merges_f)


if __name__ == "__main__":
    main()