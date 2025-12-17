#!/usr/bin/env python3
"""
Example script for evaluating Qwen2 JAX model with evalscope.

Prerequisites:
    pip install evalscope

Usage:
    1. Start the API server in another terminal:
       python -m bonsai.models.qwen2.tests.serve_api

    2. Run this evaluation script:
       python bonsai/models/qwen2/tests/run_evalscope.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import requests


def is_server_running(base_url: str = "http://localhost:8000") -> bool:
    """Check if the API server is running."""
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def run_evaluation(
    base_url: str = "http://localhost:8000",
    datasets: list[str] = None,
    limit: int = 10,
    output_dir: str = "./eval_results",
):
    """Run evalscope evaluation."""
    if datasets is None:
        datasets = ["ceval"]

    print(f"\n{'='*60}")
    print(f"Running evalscope evaluation")
    print(f"  Base URL: {base_url}")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Sample limit: {limit}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build evalscope command
    cmd = [
        "evalscope", "eval",
        "--model", "qwen2-jax",
        "--datasets", *datasets,
        "--limit", str(limit),
        "--work-dir", output_dir,
        "--api-url", f"{base_url}/v1/chat/completions",
    ]

    print(f"Running command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ Evaluation completed successfully")
        print(f"  Results saved to: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Evaluation failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("\n✗ evalscope not found. Please install it:")
        print("  pip install evalscope")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run evalscope evaluation for Qwen2 JAX")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000",
                        help="Base URL of the API server")
    parser.add_argument("--datasets", type=str, default="gsm8k",
                        help="Comma-separated list of datasets (e.g., ceval,mmlu)")
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of samples to evaluate per dataset")
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--start-server", action="store_true",
                        help="Start the API server automatically")
    parser.add_argument("--model-size", type=str, default="0.5b",
                        choices=["0.5b", "1.5b", "7b", "72b"],
                        help="Model size (only used if --start-server is set)")

    args = parser.parse_args()

    server_process = None

    try:
        # Check if server is already running
        if not is_server_running(args.base_url):
            if args.start_server:
                print(f"Starting API server...")
                server_cmd = [
                    sys.executable, "-m", "bonsai.models.qwen2.tests.serve_api",
                    "--port", args.base_url.split(":")[-1],
                    "--model-size", args.model_size,
                ]
                server_process = subprocess.Popen(
                    server_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                # Wait for server to start
                print("Waiting for server to be ready...")
                max_wait = 60
                for _ in range(max_wait):
                    if is_server_running(args.base_url):
                        print("✓ Server is ready\n")
                        break
                    time.sleep(1)
                else:
                    print("✗ Server did not start within timeout")
                    sys.exit(1)
            else:
                print(f"✗ Server is not running at {args.base_url}")
                print("\nPlease start the server first:")
                print("  python -m bonsai.models.qwen2.tests.serve_api")
                print("\nOr use --start-server flag to start it automatically")
                sys.exit(1)

        # Run evaluation
        datasets = [d.strip() for d in args.datasets.split(",")]
        success = run_evaluation(
            base_url=args.base_url,
            datasets=datasets,
            limit=args.limit,
            output_dir=args.output_dir,
        )

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    finally:
        if server_process:
            print("\nStopping server...")
            server_process.terminate()
            server_process.wait()


if __name__ == "__main__":
    main()
