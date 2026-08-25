import csv, yaml
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

fieldnames = [
    "model_size",
    "mode",
    "d_model",
    "d_ff",
    "num_layers",
    "num_heads",
    "batch_size",
    "sequence_length",
    "warmup_steps",
    "measurement_steps",
    "mean_ms",
    "std_ms",
    "status",
    "error",
]

def main():
    # 读取超参数配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_configs = config['models']
    model_size_list = list(model_configs.keys())
    modes = list(config['benchmark']['modes'])
    benchmark_config = config['benchmark']
    warmup_values = benchmark_config["warmup_values"]


    csv_path = Path(config['output']['csv_path'])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_exists = csv_path.exists() and csv_path.stat().st_size > 0

    for warmup_steps in warmup_values:
        for model_size in model_size_list:
            for mode in modes:
                command = [
                sys.executable,
                "-m",
                "cs336_systems.benchmark",
                "--config",
                args.config,
                "--model_size",
                model_size,
                "--mode",
                mode,
                "--warmup_steps",
                str(warmup_steps),
            ]

                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                print("return code:", completed.returncode)
                print("stdout:", completed.stdout)
                print("stderr:", completed.stderr)
                if completed.returncode == 0:
                    output_lines = [
                        line for line in completed.stdout.splitlines()
                        if line.strip()
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
                    last_error_line = error_lines[-1] if error_lines else "Unknown error"

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
                            "sequence_length": config["benchmark"]["sequence_length"],
                            "warmup_steps": warmup_steps,
                            "measurement_steps": config["benchmark"]["measurement_steps"],
                            "status": status,
                            "error": error_message,
                        }

                print("Parsed result:", result)
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)

                    if not csv_exists:
                        writer.writeheader()
                        csv_exists = True

                    row = {field: result.get(field, "") for field in fieldnames}
                    writer.writerow(row)

if __name__ == "__main__":
    main()