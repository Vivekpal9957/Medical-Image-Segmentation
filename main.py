"""
Main entry point for the Medical Image Segmentation Project.
Run: python main.py
"""
from train import train_model
from evaluate import evaluate_model, plot_training_history


def main():
    print("=" * 60)
    print("  MEDICAL IMAGE SEGMENTATION USING U-NET")
    print("  College Mini Project")
    print("=" * 60)
    
    # Phase 1: Training
    print("\n[PHASE 1] Training the U-Net model...")
    model, history = train_model(epochs=15, batch_size=4, lr=1e-4)
    
    # Phase 2: Plot training curves
    print("\n[PHASE 2] Generating training plots...")
    plot_training_history(history)
    
    # Phase 3: Evaluation
    print("\n[PHASE 3] Evaluating on test samples...")
    evaluate_model(num_samples=5)
    
    print("\n" + "=" * 60)
    print("  PROJECT COMPLETE!")
    print("  Check 'results/' folder for visualizations")
    print("  Check 'checkpoints/' folder for saved model")
    print("=" * 60)


if __name__ == "__main__":
    main()