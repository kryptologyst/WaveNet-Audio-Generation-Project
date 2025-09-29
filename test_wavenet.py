#!/usr/bin/env python3
"""
Test script for WaveNet Audio Generation Project
This script verifies that all components work correctly
"""

import sys
import os
import torch
import numpy as np
from wavenet_model import (
    Config, WaveNet, MockAudioDatabase, AudioDataset, 
    Trainer, AudioGenerator, AudioVisualizer, AudioProcessor
)

def test_config():
    """Test configuration setup"""
    print("🧪 Testing configuration...")
    config = Config()
    assert config.sample_rate == 16000
    assert config.quantization_channels == 256
    assert config.dilation_depth == 10
    print("✅ Configuration test passed")

def test_audio_processor():
    """Test audio processing utilities"""
    print("🧪 Testing audio processor...")
    processor = AudioProcessor()
    
    # Test quantization
    audio = np.random.uniform(-1, 1, 1000)
    quantized = processor.quantize_audio(audio)
    assert quantized.min() >= 0
    assert quantized.max() < 256
    
    # Test dequantization
    dequantized = processor.dequantize_audio(quantized)
    assert dequantized.min() >= -1
    assert dequantized.max() <= 1
    
    # Test one-hot encoding
    tensor = torch.tensor([0, 1, 2])
    one_hot = processor.one_hot_encode(tensor, 3)
    assert one_hot.shape == (3, 3)
    
    print("✅ Audio processor test passed")

def test_model():
    """Test WaveNet model"""
    print("🧪 Testing WaveNet model...")
    config = Config()
    model = WaveNet(config)
    
    # Test forward pass
    batch_size = 2
    seq_length = 1000
    input_tensor = torch.randn(batch_size, config.quantization_channels, seq_length)
    output = model(input_tensor)
    
    assert output.shape == (batch_size, config.quantization_channels, seq_length)
    
    # Test parameter count
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    print(f"   Model has {param_count:,} parameters")
    
    print("✅ WaveNet model test passed")

def test_database():
    """Test mock audio database"""
    print("🧪 Testing mock audio database...")
    config = Config()
    database = MockAudioDatabase(config)
    
    assert len(database.audio_samples) > 0
    
    # Test sample retrieval
    sample = database.get_sample(0)
    assert 'audio' in sample
    assert 'name' in sample
    assert 'type' in sample
    
    # Test random sample
    random_sample = database.get_random_sample()
    assert 'audio' in random_sample
    
    print("✅ Mock audio database test passed")

def test_dataset():
    """Test dataset and dataloader"""
    print("🧪 Testing dataset...")
    config = Config()
    database = MockAudioDatabase(config)
    dataset = AudioDataset(database, config)
    
    assert len(dataset) > 0
    
    # Test data loading
    input_seq, target_seq = dataset[0]
    assert input_seq.shape[0] == target_seq.shape[0]
    assert input_seq.shape[0] > 0
    
    # Test dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    input_batch, target_batch = batch
    assert input_batch.shape[0] == 2
    
    print("✅ Dataset test passed")

def test_training():
    """Test training functionality"""
    print("🧪 Testing training...")
    config = Config()
    database = MockAudioDatabase(config)
    dataset = AudioDataset(database, config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = WaveNet(config)
    trainer = Trainer(model, config)
    
    # Test one training step
    initial_loss = trainer.train_epoch(dataloader)
    assert initial_loss > 0
    
    print("✅ Training test passed")

def test_generation():
    """Test audio generation"""
    print("🧪 Testing audio generation...")
    config = Config()
    model = WaveNet(config)
    generator = AudioGenerator(model, config)
    
    # Generate short audio
    audio = generator.generate_audio(length=100, temperature=0.8, seed=42)
    assert len(audio) == 100
    assert audio.min() >= -1
    assert audio.max() <= 1
    
    print("✅ Audio generation test passed")

def test_visualization():
    """Test visualization utilities"""
    print("🧪 Testing visualization...")
    visualizer = AudioVisualizer()
    
    # Create test audio
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 1000))
    
    # Test plotting (without showing)
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    fig1 = visualizer.plot_waveform(audio, 16000, "Test")
    assert fig1 is not None
    
    fig2 = visualizer.plot_spectrogram(audio, 16000, "Test")
    assert fig2 is not None
    
    print("✅ Visualization test passed")

def main():
    """Run all tests"""
    print("🎵 WaveNet Audio Generation - Test Suite")
    print("=" * 50)
    
    try:
        test_config()
        test_audio_processor()
        test_model()
        test_database()
        test_dataset()
        test_training()
        test_generation()
        test_visualization()
        
        print("\n🎉 All tests passed successfully!")
        print("✅ Project is ready for use")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
