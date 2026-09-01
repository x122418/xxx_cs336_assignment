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
                if completed.returncode == 0:
                    output_lines = [
                        line for line in completed.stdout.splitlines() if line.strip()
                    ]

                    try:
                        result = json.loads(output_lines[-1])
                        result["error"] = ""
                    except (IndexError, json.JSONDecodeError) as error:
                        result = {
                            "model_size": model_size,
                            "mode": mode,
                            "status": "parse_error",
                            "error": str(error),
                        }
                else:
                    stderr_text = completed.stderr.strip()
                    error_lines = stderr_text.splitlines()
                    last_error_line = (
                        error_lines[-1] if error_lines else "Unknown error"
                    )

                    if "out of memory" in stderr_text.lower():
                        status = "oom"
                        error_message = "CUDA out of memory"
                    else:
                        status = "error"
                        error_message = last_error_line

                    failed_model_cfg = model_configs[model_size]

                    result = {
                        "model_size": model_size,
                        "mode": mode,
                        "d_model": failed_model_cfg["d_model"],
                        "d_ff": failed_model_cfg["d_ff"],
                        "num_layers": failed_model_cfg["num_layers"],
                        "num_heads": failed_model_cfg["num_heads"],
                        "batch_size": config["benchmark"]["batch_size"],
                        "sequence_length": sequence_length,
                        "measurement_steps": config["benchmark"]["measurement_steps"],
                        "status": status,
                        "error": error_message,
                    }

                print("Parsed result:", result)



if __name__ == "__main__":
    main()
