#!/usr/bin/env python3
"""
Demo script for WaveNet Audio Generation Project
This script demonstrates the key features of the project
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from wavenet_model import (
    Config, WaveNet, MockAudioDatabase, AudioDataset, 
    Trainer, AudioGenerator, AudioVisualizer, AudioProcessor
)

def demo_audio_processing():
    """Demonstrate audio processing capabilities"""
    print("🎵 Audio Processing Demo")
    print("-" * 30)
    
    processor = AudioProcessor()
    
    # Create a test audio signal
    t = np.linspace(0, 1, 16000)  # 1 second at 16kHz
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note
    
    print(f"Original audio: {len(audio)} samples, range [{audio.min():.3f}, {audio.max():.3f}]")
    
    # Quantize audio
    quantized = processor.quantize_audio(audio)
    print(f"Quantized audio: {len(quantized)} samples, range [{quantized.min()}, {quantized.max()}]")
    
    # Dequantize back
    dequantized = processor.dequantize_audio(quantized)
    print(f"Dequantized audio: range [{dequantized.min():.3f}, {dequantized.max():.3f}]")
    
    # Calculate reconstruction error
    error = np.mean((audio - dequantized) ** 2)
    print(f"Reconstruction error: {error:.6f}")
    
    print()

def demo_model_architecture():
    """Demonstrate model architecture"""
    print("🏗️ Model Architecture Demo")
    print("-" * 30)
    
    config = Config()
    model = WaveNet(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model configuration:")
    print(f"  Sample rate: {config.sample_rate} Hz")
    print(f"  Quantization channels: {config.quantization_channels}")
    print(f"  Residual channels: {config.residual_channels}")
    print(f"  Skip channels: {config.skip_channels}")
    print(f"  Dilation depth: {config.dilation_depth}")
    print(f"  Receptive field: {config.receptive_field} samples")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 1
    seq_length = 1000
    input_tensor = torch.randn(batch_size, config.quantization_channels, seq_length)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Output shape: {output.shape}")
    
    print()

def demo_training_data():
    """Demonstrate training data"""
    print("🎼 Training Data Demo")
    print("-" * 30)
    
    config = Config()
    database = MockAudioDatabase(config)
    
    print(f"Generated {len(database.audio_samples)} audio samples:")
    
    for i, sample in enumerate(database.audio_samples):
        print(f"  {i+1}. {sample['name']} ({sample['type']})")
        if 'frequency' in sample:
            print(f"     Frequency: {sample['frequency']} Hz")
        elif 'frequencies' in sample:
            print(f"     Frequencies: {sample['frequencies']} Hz")
        elif 'start_freq' in sample:
            print(f"     Frequency range: {sample['start_freq']}-{sample['end_freq']} Hz")
    
    print()

def demo_training():
    """Demonstrate training process"""
    print("🚀 Training Demo")
    print("-" * 30)
    
    config = Config()
    database = MockAudioDatabase(config)
    dataset = AudioDataset(database, config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = WaveNet(config)
    trainer = Trainer(model, config)
    
    print(f"Training on {len(dataset)} samples")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    
    # Train for a few epochs
    for epoch in range(3):
        avg_loss = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch + 1}: Average loss = {avg_loss:.4f}")
    
    print()

def demo_generation():
    """Demonstrate audio generation"""
    print("🎹 Audio Generation Demo")
    print("-" * 30)
    
    config = Config()
    model = WaveNet(config)
    generator = AudioGenerator(model, config)
    
    # Generate audio with different temperatures
    temperatures = [0.1, 0.5, 1.0, 1.5]
    length = 2000  # ~0.125 seconds at 16kHz
    
    for temp in temperatures:
        print(f"Generating audio with temperature {temp}...")
        audio = generator.generate_audio(length=length, temperature=temp, seed=42)
        
        # Calculate statistics
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        
        print(f"  Length: {len(audio)} samples")
        print(f"  RMS: {rms:.4f}")
        print(f"  Peak: {peak:.4f}")
        print(f"  Range: [{audio.min():.4f}, {audio.max():.4f}]")
    
    print()

def demo_visualization():
    """Demonstrate visualization capabilities"""
    print("📊 Visualization Demo")
    print("-" * 30)
    
    config = Config()
    database = MockAudioDatabase(config)
    visualizer = AudioVisualizer()
    
    # Get a sample
    sample = database.get_sample(0)
    audio = sample['audio']
    
    print(f"Visualizing: {sample['name']}")
    print(f"Audio length: {len(audio)} samples")
    print(f"Duration: {len(audio) / config.sample_rate:.2f} seconds")
    
    # Create plots
    fig1 = visualizer.plot_waveform(audio, config.sample_rate, f"{sample['name']} Waveform")
    fig2 = visualizer.plot_spectrogram(audio, config.sample_rate, f"{sample['name']} Spectrogram")
    
    print("Plots created successfully!")
    print()

def main():
    """Run the complete demo"""
    print("🎵 WaveNet Audio Generation - Complete Demo")
    print("=" * 60)
    print()
    
    # Run all demos
    demo_audio_processing()
    demo_model_architecture()
    demo_training_data()
    demo_training()
    demo_generation()
    demo_visualization()
    
    print("🎉 Demo completed successfully!")
    print()
    print("Next steps:")
    print("1. Run 'python 0137.py' for the full training and generation")
    print("2. Run 'streamlit run app.py' for the web interface")
    print("3. Run 'python test_wavenet.py' to verify everything works")

if __name__ == "__main__":
    main()
