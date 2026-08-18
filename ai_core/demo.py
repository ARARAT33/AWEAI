"""
AI Core Framework - Comprehensive Demo Script
Demonstrates all components working together in a real-world scenario.
"""

import numpy as np
from ai_core import (
    NeuralEngine, LayerConfig, ActivationType,
    AdaptiveLearner, LearningStrategy,
    DataProcessor, FeatureExtractor,
    OptimizationEngine, GradientOptimizer,
    TransformerCore, AttentionMechanism,
    MemoryManager, CognitiveBuffer,
    initialize_framework
)


def demo_neural_engine():
    """Demonstrate neural network training."""
    print("\n" + "="*60)
    print("🧠 NEURAL ENGINE DEMO")
    print("="*60)
    
    # Create a simple classification network
    layer_configs = [
        LayerConfig(10, 32, ActivationType.RELU, dropout_rate=0.1),
        LayerConfig(32, 16, ActivationType.GELU, dropout_rate=0.1),
        LayerConfig(16, 3, ActivationType.SOFTMAX)  # 3 classes
    ]
    
    model = NeuralEngine(layer_configs, learning_rate=0.01)
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y_true = np.random.randint(0, 3, 1000)
    y_onehot = np.zeros((1000, 3))
    y_onehot[np.arange(1000), y_true] = 1
    
    # Train for a few epochs
    print("\nTraining neural network...")
    for epoch in range(5):
        loss = model.train_epoch(X, y_onehot, batch_size=32)
        metrics = model.evaluate(X, y_onehot)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={metrics['accuracy']:.4f}")
    
    arch = model.get_architecture()
    print(f"\nArchitecture: {arch['layers'][0]['input_size']} → "
          f"{arch['layers'][0]['output_size']} → "
          f"{arch['layers'][-1]['output_size']}")
    print(f"Total parameters: {arch['total_parameters']}")
    
    return model


def demo_adaptive_learner():
    """Demonstrate adaptive learning strategies."""
    print("\n" + "="*60)
    print("⚡ ADAPTIVE LEARNER DEMO")
    print("="*60)
    
    learner = AdaptiveLearner(
        strategy=LearningStrategy.ADAM,
        base_learning_rate=0.001,
        weight_decay=0.01,
        warmup_steps=100
    )
    
    # Simulate optimization steps
    params = np.random.randn(100)
    print("\nOptimizing with Adam optimizer...")
    
    for step in range(10):
        gradients = np.random.randn(100) * 0.1
        params = learner.optimize(params, gradients)
        lr = learner.current_lr
        
        if step % 3 == 0:
            print(f"Step {step}: LR={lr:.6f}, Param norm={np.linalg.norm(params):.4f}")
    
    state = learner.get_optimization_state()
    print(f"\nFinal state: {state['strategy']}, avg_loss={state['avg_loss']:.4f}" if state['avg_loss'] else "")


def demo_transformer():
    """Demonstrate transformer architecture."""
    print("\n" + "="*60)
    print("🔄 TRANSFORMER CORE DEMO")
    print("="*60)
    
    # Create a small transformer
    transformer = TransformerCore(
        vocab_size=1000,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout_rate=0.1
    )
    
    # Generate sample input
    batch_size = 4
    seq_length = 20
    input_ids = np.random.randint(0, 1000, (batch_size, seq_length))
    
    print(f"\nTransformer config:")
    print(f"  Vocab size: {transformer.vocab_size}")
    print(f"  Embed dim: {transformer.embed_dim}")
    print(f"  Num heads: {transformer.num_heads}")
    print(f"  Num layers: {transformer.num_layers}")
    print(f"  Total parameters: {transformer.count_parameters()}")
    
    # Forward pass
    encoder_output = transformer.encode(input_ids, training=False)
    print(f"\nEncoder output shape: {encoder_output.shape}")
    
    # Full forward pass
    logits = transformer.forward(input_ids)
    print(f"Output logits shape: {logits.shape}")


def demo_data_processor():
    """Demonstrate data processing pipeline."""
    print("\n" + "="*60)
    print("📊 DATA PROCESSOR DEMO")
    print("="*60)
    
    processor = DataProcessor(
        normalization='zscore',
        augmentation_enabled=True,
        batch_size=32,
        validation_split=0.2
    )
    
    # Generate sample data
    X = np.random.randn(500, 20) * 10 + 5
    y = np.random.randint(0, 3, 500)
    
    print(f"\nOriginal data: mean={X.mean():.2f}, std={X.std():.2f}")
    
    # Process data
    X_processed, y_processed = processor.process_pipeline(X, y, fit=True)
    print(f"Processed data: mean={X_processed.mean():.4f}, std={X_processed.std():.4f}")
    
    # Create train/val split
    X_train, X_val, y_train, y_val = processor.create_train_val_split(X_processed, y_processed)
    print(f"\nTrain set: {X_train.shape}, Val set: {X_val.shape}")
    
    # Compute class weights
    class_weights = processor.compute_class_weights(y_train)
    print(f"Class weights: {class_weights}")
    
    # Extract features
    extractor = FeatureExtractor(feature_types=['statistical', 'temporal'])
    features = extractor.fit_transform(X[:10])
    print(f"Extracted features shape: {features.shape}")


def demo_memory_system():
    """Demonstrate memory management system."""
    print("\n" + "="*60)
    print("💾 MEMORY MANAGER DEMO")
    print("="*60)
    
    memory = MemoryManager(
        working_memory_capacity=50,
        long_term_memory_capacity=500
    )
    
    # Store different types of memories
    print("\nStoring memories...")
    
    # Episodic memory
    episode1 = {'event': 'User asked about AI', 'context': 'conversation', 'timestamp': 1}
    episode2 = {'event': 'Model trained successfully', 'context': 'training', 'timestamp': 2}
    memory.store_episode(episode1)
    memory.store_episode(episode2)
    
    # Semantic memory
    memory.store_semantic_fact('neural_network', 
                               'A computational model inspired by biological neurons',
                               ['AI', 'machine_learning', 'deep_learning'])
    
    memory.store_semantic_fact('transformer',
                              'Attention-based architecture for sequence processing',
                              ['NLP', 'attention', 'deep_learning'])
    
    # Search memories
    print("\nSearching for 'AI' related memories...")
    results = memory.retrieve_relevant('AI', top_k=3)
    
    for mem_type, items in results.items():
        if items:
            print(f"  {mem_type}: {len(items)} results")
    
    # Run consolidation
    consolidated = memory.consolidate_to_long_term(min_priority=0.5)
    print(f"\nConsolidated {consolidated} memories to long-term storage")
    
    # Get statistics
    stats = memory.get_comprehensive_statistics()
    print(f"\nMemory stats:")
    print(f"  Working memory: {stats['working_memory']['total_memories']} items")
    print(f"  Long-term memory: {stats['long_term_memory']['total_memories']} items")
    print(f"  Episodic memories: {stats['episodic_memory_size']}")
    print(f"  Semantic concepts: {stats['semantic_memory_size']}")


def demo_optimization_engine():
    """Demonstrate advanced optimization."""
    print("\n" + "="*60)
    print("🎯 OPTIMIZATION ENGINE DEMO")
    print("="*60)
    
    opt_engine = OptimizationEngine(
        optimizer_config={
            'optimizer_type': 'adamw',
            'learning_rate': 0.001,
            'weight_decay': 0.01
        },
        early_stopping_patience=5
    )
    
    # Simulate training loop
    print("\nSimulating training with optimization monitoring...")
    
    losses = []
    for epoch in range(15):
        # Simulate decreasing loss with noise
        loss = 1.0 / (1 + epoch * 0.3) + np.random.randn() * 0.05
        losses.append(loss)
        
        should_stop = opt_engine.check_early_stopping(loss)
        
        if epoch % 3 == 0:
            diagnostics = opt_engine.get_training_diagnostics()
            print(f"Epoch {epoch}: Loss={loss:.4f}, Best={diagnostics['best_loss']:.4f}")
        
        if should_stop:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Check convergence
    converged = opt_engine.check_convergence()
    print(f"\nConvergence status: {converged}")
    
    final_diagnostics = opt_engine.get_training_diagnostics()
    print(f"Final loss trend: {final_diagnostics.get('loss_trend', 0):.6f}")


def main():
    """Run all demos."""
    print("\n" + "🚀"*30)
    print("   AI CORE FRAMEWORK - COMPREHENSIVE DEMO")
    print("🚀"*30)
    
    # Initialize framework
    framework_info = initialize_framework(verbose=True)
    
    # Run all demonstrations
    demo_neural_engine()
    demo_adaptive_learner()
    demo_transformer()
    demo_data_processor()
    demo_memory_system()
    demo_optimization_engine()
    
    print("\n" + "="*60)
    print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nThe AI Core Framework provides:")
    print("  • Advanced neural network engines")
    print("  • Adaptive learning strategies")
    print("  • Transformer architectures")
    print("  • Comprehensive data processing")
    print("  • Cognitive memory systems")
    print("  • State-of-the-art optimization")
    print("\nReady for production AI applications! 🎉\n")


if __name__ == "__main__":
    main()
