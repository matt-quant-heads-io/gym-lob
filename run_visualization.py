"""
Run the DOM visualization GUI with a trained model
"""

import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui import InteractiveEvaluator

def main():
    parser = argparse.ArgumentParser(description='Run DOM Visualization GUI')
    parser.add_argument('--model', default='trained_model.pth', 
                       help='Path to trained model')
    parser.add_argument('--config', default='config/default.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Please train a model first by running: python train.py")
        return
    
    # Create and run evaluator
    evaluator = InteractiveEvaluator(args.model, args.config)
    evaluator.run()

if __name__ == "__main__":
    main()