import argparse
import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).parent
sys.path.append(str(root))

from experiment.run_comprehensive_benchmark import run_benchmark
from experiment.run_predictors import main as run_predictors_analysis

def main():
    parser = argparse.ArgumentParser(description="EduRuleReasoning Framework Entry Point")
    parser.add_argument('mode', choices=['benchmark', 'analysis'], help='Mode to run: "benchmark" for all models comparison, "analysis" for deep predictor detailed analysis')
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    if args.mode == 'benchmark':
        print("Starting Comprehensive Benchmark...")
        run_benchmark()
    elif args.mode == 'analysis':
        print("Starting Deep Predictor & F-Logic Analysis...")
        run_predictors_analysis()

if __name__ == "__main__":
    main()
