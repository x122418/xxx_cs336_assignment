import yaml, argparse
import numpy as np
import torch
import random
import json
import time
from pathlib import Path
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter

from hw2.transformer import Transformer_LM

from hw3.lr_schedule import learning_rate_schedule, grad_clip
from hw3.AdamW import AdamW
from hw3.cross_entropy import cross_entropy

from hw4.data_loading import data_loading
from hw4.check_point import load_checkpoint, save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()


def main():
    # 读取超参数配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    experiment_cfg = config["experiment"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    runtime_cfg = config["runtime"]
    checkpoint_cfg = config["checkpoint"]

    checkpoint_dir = Path(checkpoint_cfg["output_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(
    log_dir=checkpoint_dir / "tensorboard"
)

    # 设置random seed和device
    seed = experiment_cfg["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(runtime_cfg["device"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Using device: {device}")

    # mmap读取tokenized后的数据
    train_data_path = data_cfg["train_path"]
    val_data_path = data_cfg["val_path"]

    train_data = np.load(train_data_path, mmap_mode="r")
    val_data = np.load(val_data_path, mmap_mode="r")

    batch_size = training_cfg["batch_size"]
    train_context_length = training_cfg["train_context_length"]

    # 构建语言model实例
    vocab_size = model_cfg["vocab_size"]
    d_model = model_cfg["d_model"]
    num_heads = model_cfg["num_heads"]
    max_seq_len = model_cfg["max_seq_len"]

    assert train_data.ndim == 1, "train_data must be 1D"
    assert val_data.ndim == 1, "val_data must be 1D"

    assert batch_size > 0
    assert 0 < train_context_length <= max_seq_len

    assert len(train_data) > train_context_length
    assert len(val_data) > train_context_length

    assert training_cfg["max_iters"] > 0
    assert training_cfg["warmup_iters"] >= 0
    assert (
        training_cfg["cosine_cycle_iters"]
        > training_cfg["warmup_iters"]
    )

    assert checkpoint_cfg["log_interval"] > 0
    assert checkpoint_cfg["eval_interval"] > 0
    assert checkpoint_cfg["eval_batches"] > 0
    assert checkpoint_cfg["save_interval"] > 0

    assert train_data.min() >= 0
    assert val_data.min() >= 0

    assert train_data.max() < vocab_size
    assert val_data.max() < vocab_size

    if device.type == "cuda":
        assert torch.cuda.is_available(), (
            "CUDA device requested, but CUDA is unavailable"
        )


    lm_model = Transformer_LM(
        vocab_size,
        max_seq_len,
        d_model,
        num_heads,
        model_cfg["d_ff"],
        model_cfg["num_layers"],
        model_cfg["theta"],
    ).to(device)

    # 构建优化器实例
    optimizer = AdamW(
        lm_model.parameters(),
        lr=training_cfg["max_lr"],
        betas=tuple(training_cfg["betas"]),
        eps=training_cfg["eps"],
        weight_decay=training_cfg["weight_decay"],
    )

    # 恢复checkpoint状态
    start_iteration = 0
    assert 0 <= start_iteration <= training_cfg["max_iters"]
    resume_from = checkpoint_cfg["resume_from"]
    if resume_from is not None:
        start_iteration = load_checkpoint(
            src=resume_from, model=lm_model, optimizer=optimizer
        )

    # training loop
    max_iters = training_cfg["max_iters"]
    log_path = checkpoint_dir / "metrics.jsonl"
    training_start_time = time.perf_counter()
    lm_model.train()

    for iteration in range(start_iteration, max_iters):

        # 更新学习率
        current_lr = learning_rate_schedule(
            iteration,
            training_cfg["max_lr"],
            training_cfg["min_lr"],
            training_cfg["warmup_iters"],
            training_cfg["cosine_cycle_iters"],                                                       
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        # 加载batch级别数据
        train_inputs, train_targets = data_loading(
            train_data, batch_size, train_context_length, device
        )

        # 清除上一轮梯度
        optimizer.zero_grad()

        # 正向传播 得到logits
        logits = lm_model(train_inputs)

        # 计算loss
        flat_train_logits = rearrange(logits, "B S V -> (B S) V")
        flat_train_targets = rearrange(train_targets, "B S -> (B S)")
        train_loss = cross_entropy(flat_train_logits, flat_train_targets)

        # 反向传播（计算梯度）
        train_loss.backward()

        # 梯度裁剪
        grad_clip(lm_model.parameters(), training_cfg["max_l2_norm"])

        # optimizer 更新参数
        optimizer.step()

        # train指标 定期日志
        if (iteration + 1) % checkpoint_cfg["log_interval"] == 0:
            train_record = {
                "step": iteration + 1,
                "split": "train",
                "current_lr": current_lr,
                "loss": train_loss.item(),
                "elapsed seconds": (time.perf_counter() - training_start_time),
            }
            with log_path.open('a', encoding="utf-8") as log_file:
                log_file.write(
                    json.dumps(train_record) + "\n"
                )
            writer.add_scalar(
                "Loss/train",
                train_loss.item(),
                iteration+1,
            )
            writer.add_scalar(
                "LearningRate",
                current_lr,
                iteration + 1,
            )

            print(
                f"step = {iteration+1} "
                f"train_loss = {train_loss.item():.4f} "
                f"lr={current_lr:.6e}"
            )


        # val指标 定期输出
        if (iteration + 1) % checkpoint_cfg["eval_interval"] == 0:
            lm_model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(checkpoint_cfg["eval_batches"]):
                    val_inputs, val_targets = data_loading(
                        val_data, batch_size, train_context_length, device
                    )
                    logits = lm_model(val_inputs)

                    flat_val_logits = rearrange(logits, "B S V -> (B S) V")
                    flat_val_targets = rearrange(val_targets, "B S -> (B S)")
                    val_loss = cross_entropy(flat_val_logits, flat_val_targets)
                    val_losses.append(val_loss.item())
            mean_val_loss = sum(val_losses) / len(val_losses)
            writer.add_scalar("Loss/validation",
                              mean_val_loss, 
                              iteration + 1)
            print(f"step = {iteration+1}" f"val_loss = {mean_val_loss:.4f}")
            val_record = {
                            "step": iteration + 1,
                            "split": "val",
                            "current_lr": current_lr,
                            "loss": mean_val_loss,
                            "elapsed seconds": (time.perf_counter() - training_start_time),
                        }
            with log_path.open('a', encoding="utf-8") as log_file:
                log_file.write(
                    json.dumps(val_record) + "\n"
                            )
            lm_model.train()

        # 定期保存模型
        if (iteration + 1) % checkpoint_cfg["save_interval"] == 0:
            checkpoint_path = checkpoint_dir / f"step_{iteration + 1}.pt"
            save_checkpoint(lm_model, optimizer, iteration + 1, checkpoint_path)

    # 由于保存时定期的 可能会漏最后几个step的保存 需要额外保存
    final_checkpoint_path = checkpoint_dir / f"step_{max_iters}.pt"
    save_checkpoint(lm_model, optimizer, max_iters, final_checkpoint_path)
    print(f"Training finished. " f"Final checkpoint saved to {final_checkpoint_path}")
    writer.close()


if __name__ == "__main__":
    main()
