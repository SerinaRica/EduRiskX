
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
root = Path(__file__).parent.parent
sys.path.append(str(root))

from experiment.run_weekly_benchmark import WeeklyBenchmarkRunner

def main():
    print("Updating Early Detection Analysis...")
    
    runner = WeeklyBenchmarkRunner(
        data_path=root / "data/processed/weekly_features.parquet",
        output_dir=root / "outputs/experiments/weekly_benchmark"
    )
    
    # This will read all prediction files in predictions/ and update early_detection_analysis.csv
    runner.analyze_early_detection()
    
    print("Early detection analysis updated.")

if __name__ == "__main__":
    main()
