"""Train the same tiny model with each attention variant swapped in, log
perplexity / step time / peak memory to results.csv, and render charts.

Usage:
    python run_comparison.py --variants self causal multihead gqa mqa
"""
import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", required=True)
    args = parser.parse_args()

    results = []
    for variant in args.variants:
        # TODO: build model with this variant's attention, train briefly, record metrics
        results.append({"variant": variant, "perplexity": None, "step_time_ms": None, "peak_mem_mb": None})

    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "perplexity", "step_time_ms", "peak_mem_mb"])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
