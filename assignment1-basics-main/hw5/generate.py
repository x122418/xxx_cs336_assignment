from pathlib import Path
import yaml, argparse, torch
from hw2.transformer import Transformer_LM
from hw1.tokenizer_v2 import Tokenizer
from hw5.decoding import decode_from_llm
from hw4.check_point import load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()


def main():
    # 读取超参数配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)


    model_cfg = config["model"]
    runtime_cfg = config["runtime"]
    generation_cfg = config["generation"]
    tokenizer_cfg = config["tokenizer"]

    vocab_file_path = tokenizer_cfg['vocab_path']
    merges_file_path = tokenizer_cfg['merges_path']
    special_tokens = tokenizer_cfg['special_tokens']
    checkpoint_path = generation_cfg['checkpoint_path']

    device = torch.device(runtime_cfg['device'])
    tokenizer = Tokenizer.from_files(vocab_file_path, merges_file_path, special_tokens)
    model = Transformer_LM(
    model_cfg["vocab_size"],
    model_cfg["max_seq_len"],
    model_cfg["d_model"],
    model_cfg["num_heads"],
    model_cfg["d_ff"],
    model_cfg["num_layers"],
    model_cfg["theta"],
).to(device)

    checkpoint = torch.load(
        checkpoint_path, 
        map_location=device
    )
    model.load_state_dict(checkpoint['model'])
    print(f'Loaded checkpoint at step {checkpoint["iteration"]}')

    response = decode_from_llm(
    model=model,
    tokenizer=tokenizer,
    prompt=generation_cfg["prompt"],
    max_new_tokens=generation_cfg["max_new_tokens"],
    context_length=model_cfg["max_seq_len"],
    temperature=generation_cfg["temperature"],
    top_p=generation_cfg["top_p"],
)

    print(response)
if __name__ == "__main__":
    main()