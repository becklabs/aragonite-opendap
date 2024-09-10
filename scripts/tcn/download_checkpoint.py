import wandb
import os
import shutil
import tempfile
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Download a checkpoint from a Weights & Biases run."
    )
    parser.add_argument(
        "--project_name", default="tcn_sample", help="Name of the W&B project"
    )
    parser.add_argument(
        "--run_path",
        default="becklabs-northeastern-university/tcn_sample/gh75rf3a",
        help="Path to the W&B run",
    )
    parser.add_argument(
        "--checkpoint_name",
        default="model_epoch_95.pth",
        help="Name of the checkpoint file",
    )
    parser.add_argument(
        "--output_path",
        default="checkpoints/TCN/temperature/",
        help="Output path for the checkpoint",
    )
    return parser.parse_args()


args = parse_arguments()


api = wandb.Api()
run = api.run(args.run_path)

with tempfile.TemporaryDirectory() as temp_dir:
    checkpoint_path = (
        run.file(f"training_checkpoints/{args.project_name}/{args.checkpoint_name}")
        .download(replace=True, root=temp_dir)
        .name
    )

    os.makedirs(args.output_path, exist_ok=True)
    shutil.move(checkpoint_path, os.path.join(args.output_path, args.checkpoint_name))
