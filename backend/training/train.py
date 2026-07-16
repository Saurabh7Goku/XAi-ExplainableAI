import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.core.training import TrainingPipeline
from app.config import settings, ModelConfig
from app.utils.logger import logger


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Mango Leaf Disease Detection Model")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data", 
                       help="Directory containing the training dataset")
    parser.add_argument("--test-dir", type=str, default=None,
                       help="Directory containing test dataset for evaluation")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=ModelConfig.EPOCHS,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=ModelConfig.BATCH_SIZE,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=ModelConfig.LEARNING_RATE,
                       help="Learning rate")
    parser.add_argument("--validation-split", type=float, default=0.2,
                       help="Validation split fraction")
    
    # Model arguments
    parser.add_argument("--num-classes", type=int, default=settings.num_classes,
                       help="Number of output classes")
    parser.add_argument("--embed-dim", type=int, default=settings.embed_dim,
                       help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=settings.depth,
                       help="Number of transformer blocks")
    parser.add_argument("--num-heads", type=int, default=settings.num_heads,
                       help="Number of attention heads")
    
    # Training control arguments
    parser.add_argument("--save-best-only", action="store_true", default=True,
                       help="Save only the best model")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Directory to save trained models")
    parser.add_argument("--experiment-name", type=str, default="vit_mango_v1",
                       help="Name for this training experiment")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Starting training experiment: {args.experiment_name}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize training pipeline
    training_pipeline = TrainingPipeline()
    
    try:
        # Train the model
        results = training_pipeline.train(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_split=args.validation_split,
            save_best_only=args.save_best_only,
            early_stopping_patience=args.early_stopping_patience
        )
        
        if results['success']:
            logger.info("Training completed successfully!")
            logger.info(f"Best validation accuracy: {results['best_validation_accuracy']:.2f}%")
            logger.info(f"Training time: {results['training_time_minutes']:.2f} minutes")
            
            # Evaluate on test set if provided
            if args.test_dir and os.path.exists(args.test_dir):
                logger.info("Evaluating on test set...")
                test_results = training_pipeline.evaluate_model(
                    test_dir=args.test_dir,
                    model_path=results.get('model_path')
                )
                
                logger.info(f"Test accuracy: {test_results['overall_accuracy']:.2f}%")
                
                # Save results
                import json
                results_file = output_dir / f"{args.experiment_name}_results.json"
                with open(results_file, 'w') as f:
                    json.dump({
                        'training': results,
                        'test': test_results,
                        'arguments': vars(args)
                    }, f, indent=2)
                
                logger.info(f"Results saved to: {results_file}")
        else:
            logger.error(f"Training failed: {results.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Training failed with exception: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
