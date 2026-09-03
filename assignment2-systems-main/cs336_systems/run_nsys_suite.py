import csv, yaml
import argparse
import json
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

fieldnames = [
    "model_size",
    "sequence_length",
    "mode",
    "d_ff",
    "num_layers",
    "num_heads",
    "batch_size",
    "warmup_steps",
    "measurement_steps",
    "status",
    "error",
]


def main():
    # 读取超参数配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_configs = config["models"]
    model_size_list = list(model_configs.keys())
    modes = list(config["benchmark"]["modes"])
    benchmark_config = config["benchmark"]
    sequence_lengths = benchmark_config["sequence_lengths"]
    warmup_steps = benchmark_config["warmup_steps"]

    base_output_dir = Path(config["output"]["nsys_directory"])

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    env = os.environ.copy()
    env["NSYS_NVTX_PROFILER_REGISTER_ONLY"] = "0"
    for sequence_length in sequence_lengths:
        for model_size in model_size_list:
            for mode in modes:
                experiment_name = f"{model_size}_{sequence_length}_{mode}"
                output_stem = run_dir / experiment_name


                command = [
                    "nsys",
                    "profile",
                    "--trace=cuda,nvtx",
                    "--capture-range=nvtx",
                    "--nvtx-capture=profile_region@*",
                    "--capture-range-end=stop",       # 新增
                    "--force-overwrite=true",
                    f"--output={output_stem}",
                    
                    sys.executable,
                    "-m",
                    "cs336_systems.benchmark",
                    "--config",
                    args.config,
                    "--model_size",
                    model_size,
                    "--mode",
                    mode,
                    "--sequence_length",
                    str(sequence_length),
                    "--warmup_steps",
                    str(warmup_steps),
                    "--annotate_attention",
                ]

                completed = subprocess.run(
                    command,
                    env = env,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                print("return code:", completed.returncode)
                print("stdout:", completed.stdout)
                print("stderr:", completed.stderr)
                report_path = Path(f"{output_stem}.nsys-rep")
                if completed.returncode == 0 and report_path.exists():
                    status = "ok"
                    error_message = ""

                elif completed.returncode == 0:
                    status = "missing_report"
                    error_message = "nsys 正常退出，但没有生成报告"

                else:
                    stderr_text = completed.stderr.strip()
                    error_lines = stderr_text.splitlines()

                    if "out of memory" in stderr_text.lower():
                        status = "oom"
                        error_message = "CUDA out of memory"
                    else:
                        status = "error"
                        error_message = error_lines[-1] if error_lines else "unknown error"

                result = {
                    "model_size": model_size,
                    "mode": mode,
                    "sequence_length": sequence_length,
                    "warmup_steps": warmup_steps,
                    "status": status,
                    "report_path": str(report_path) if report_path.exists() else "",
                    "error": error_message,
                }

                print("Parsed result:", result)



if __name__ == "__main__":
    main()
