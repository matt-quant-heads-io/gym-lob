#!/usr/bin/env python3
"""
Run the DOM visualization with HTML/Browser interface
"""

import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='Run DOM Visualization')
    parser.add_argument('--model', default='trained_model.pth', 
                       help='Path to trained model')
    parser.add_argument('--config', default='config/default.yaml',
                       help='Path to config file')
    parser.add_argument('--port', type=int, default=8080,
                       help='Port for web server (default: 8080)')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of episodes to run')
    parser.add_argument('--step-delay', type=float, default=0.001,
                       help='Delay between steps in seconds')
    parser.add_argument('--headless', action='store_true',
                       help='Run in terminal mode (no browser)')
    parser.add_argument('--no-browser', action='store_true',
                       help='Start server but don\'t open browser automatically')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("Please train a model first by running: python train.py")
        return
    
    if args.headless:
        print("📊 Running in terminal mode...")
        from gui import HeadlessEvaluator
        evaluator = HeadlessEvaluator(args.model, args.config)
        evaluator.run(episodes=args.episodes, step_delay=args.step_delay)
    else:
        print("🌐 Starting web-based DOM visualization...")
        from gui import WebVisualizer
        visualizer = WebVisualizer(args.model, args.config, port=args.port)
        visualizer.run(
            episodes=args.episodes, 
            step_delay=args.step_delay,
            open_browser=not args.no_browser
        )

if __name__ == "__main__":
    main()