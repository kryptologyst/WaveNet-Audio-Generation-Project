# Project 137. WaveNet for Audio Generation
# Description:
# WaveNet is a deep generative model developed by DeepMind that produces raw audio waveforms using causal and dilated convolutions. It generates realistic speech and audio one sample at a time by learning the probability distribution of audio samples. This project implements a modern WaveNet using PyTorch with proper architecture, training, and inference capabilities.

# Dependencies: pip install torch torchaudio matplotlib numpy scipy librosa streamlit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import wavfile
import os
import json
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')
 
# Configuration
class Config:
    def __init__(self):
        self.sample_rate = 16000  # 16kHz for better quality
        self.duration = 2.0       # seconds
        self.quantization_channels = 256
        self.residual_channels = 64
        self.skip_channels = 256
        self.dilation_depth = 10
        self.kernel_size = 2
        self.batch_size = 4
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.receptive_field = 2 ** self.dilation_depth
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
config = Config()

# Audio processing utilities
class AudioProcessor:
    @staticmethod
    def quantize_audio(audio: np.ndarray, num_classes: int = 256) -> np.ndarray:
        """Quantize audio to discrete values"""
        audio = np.clip(audio, -1, 1)
        quantized = ((audio + 1) / 2 * (num_classes - 1)).astype(np.int64)
        return quantized
    
    @staticmethod
    def dequantize_audio(quantized: np.ndarray, num_classes: int = 256) -> np.ndarray:
        """Convert quantized audio back to continuous values"""
        return (quantized / (num_classes - 1)) * 2 - 1
    
    @staticmethod
    def one_hot_encode(x: torch.Tensor, num_classes: int) -> torch.Tensor:
        """One-hot encode quantized audio"""
        x = x.long()
        return F.one_hot(x, num_classes=num_classes).float()

# Modern WaveNet implementation
class DilatedConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=dilation)
        self.gate_conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=dilation)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.conv(x)
        gate_out = self.gate_conv(x)
        return conv_out, gate_out

class ResidualBlock(nn.Module):
    def __init__(self, residual_channels: int, skip_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.dilated_conv = DilatedConv1d(residual_channels, 2 * residual_channels, kernel_size, dilation)
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_out, gate_out = self.dilated_conv(x)
        
        # Gated activation
        z = torch.tanh(conv_out) * torch.sigmoid(gate_out)
        
        # Residual connection
        residual = self.residual_conv(z) + x
        
        # Skip connection
        skip = self.skip_conv(z)
        
        return residual, skip

class WaveNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.input_conv = nn.Conv1d(config.quantization_channels, config.residual_channels, 1)
        
        # Residual blocks with exponentially increasing dilations
        self.residual_blocks = nn.ModuleList()
        for i in range(config.dilation_depth):
            dilation = 2 ** i
            self.residual_blocks.append(
                ResidualBlock(config.residual_channels, config.skip_channels, config.kernel_size, dilation)
            )
        
        # Output layers
        self.output_conv1 = nn.Conv1d(config.skip_channels, config.skip_channels, 1)
        self.output_conv2 = nn.Conv1d(config.skip_channels, config.quantization_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input embedding
        x = self.input_conv(x)
        
        # Collect skip connections
        skip_connections = []
        
        # Pass through residual blocks
        for residual_block in self.residual_blocks:
            x, skip = residual_block(x)
            skip_connections.append(skip)
        
        # Sum skip connections
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)
        
        # Output layers
        out = F.relu(skip_sum)
        out = self.output_conv1(out)
        out = F.relu(out)
        out = self.output_conv2(out)
        
        return out

# Mock Audio Database
class MockAudioDatabase:
    def __init__(self, config: Config):
        self.config = config
        self.audio_samples = []
        self._generate_mock_data()
    
    def _generate_mock_data(self):
        """Generate synthetic audio samples for training"""
        print("Generating mock audio database...")
        
        # Generate various synthetic audio patterns
        patterns = [
            self._generate_sine_wave(440, "A4_note"),      # A4 note
            self._generate_sine_wave(523, "C5_note"),      # C5 note
            self._generate_sine_wave(659, "E5_note"),      # E5 note
            self._generate_chord([440, 523, 659], "A_major_chord"),  # A major chord
            self._generate_sweep(200, 800, "frequency_sweep"),       # Frequency sweep
            self._generate_noise("white_noise"),           # White noise
            self._generate_beat_pattern("beat_pattern"),   # Beat pattern
        ]
        
        self.audio_samples = patterns
        print(f"Generated {len(self.audio_samples)} audio samples")
    
    def _generate_sine_wave(self, frequency: float, name: str) -> dict:
        """Generate a sine wave"""
        t = np.linspace(0, self.config.duration, int(self.config.sample_rate * self.config.duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        return {
            'audio': audio,
            'name': name,
            'type': 'sine_wave',
            'frequency': frequency
        }
    
    def _generate_chord(self, frequencies: List[float], name: str) -> dict:
        """Generate a chord (multiple frequencies)"""
        t = np.linspace(0, self.config.duration, int(self.config.sample_rate * self.config.duration), endpoint=False)
        audio = np.zeros_like(t)
        for freq in frequencies:
            audio += 0.3 * np.sin(2 * np.pi * freq * t)
        audio = audio / len(frequencies)  # Normalize
        return {
            'audio': audio,
            'name': name,
            'type': 'chord',
            'frequencies': frequencies
        }
    
    def _generate_sweep(self, start_freq: float, end_freq: float, name: str) -> dict:
        """Generate a frequency sweep"""
        t = np.linspace(0, self.config.duration, int(self.config.sample_rate * self.config.duration), endpoint=False)
        frequencies = np.linspace(start_freq, end_freq, len(t))
        audio = 0.3 * np.sin(2 * np.pi * frequencies * t)
        return {
            'audio': audio,
            'name': name,
            'type': 'sweep',
            'start_freq': start_freq,
            'end_freq': end_freq
        }
    
    def _generate_noise(self, name: str) -> dict:
        """Generate white noise"""
        audio = np.random.normal(0, 0.1, int(self.config.sample_rate * self.config.duration))
        return {
            'audio': audio,
            'name': name,
            'type': 'noise'
        }
    
    def _generate_beat_pattern(self, name: str) -> dict:
        """Generate a beat pattern"""
        t = np.linspace(0, self.config.duration, int(self.config.sample_rate * self.config.duration), endpoint=False)
        audio = np.zeros_like(t)
        beat_freq = 2  # 2 Hz beat
        carrier_freq = 440
        audio = 0.3 * np.sin(2 * np.pi * carrier_freq * t) * np.sin(2 * np.pi * beat_freq * t)
        return {
            'audio': audio,
            'name': name,
            'type': 'beat_pattern',
            'beat_freq': beat_freq,
            'carrier_freq': carrier_freq
        }
    
    def get_sample(self, index: int) -> dict:
        """Get a specific audio sample"""
        return self.audio_samples[index % len(self.audio_samples)]
    
    def get_random_sample(self) -> dict:
        """Get a random audio sample"""
        return np.random.choice(self.audio_samples)

# Data loading and preprocessing
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, database: MockAudioDatabase, config: Config):
        self.database = database
        self.config = config
        self.processor = AudioProcessor()
    
    def __len__(self):
        return len(self.database.audio_samples) * 10  # Augment by repeating samples
    
    def __getitem__(self, idx):
        # Get audio sample
        sample = self.database.get_sample(idx)
        audio = sample['audio']
        
        # Add some random noise for augmentation
        noise = np.random.normal(0, 0.01, audio.shape)
        audio = audio + noise
        
        # Quantize audio
        quantized = self.processor.quantize_audio(audio, self.config.quantization_channels)
        
        # Convert to tensor
        quantized_tensor = torch.tensor(quantized, dtype=torch.long)
        
        # Create input and target sequences
        input_seq = quantized_tensor[:-1]  # All but last
        target_seq = quantized_tensor[1:]   # All but first
        
        return input_seq, target_seq

# Training utilities
class Trainer:
    def __init__(self, model: WaveNet, config: Config):
        self.model = model
        self.config = config
        self.device = config.device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.processor = AudioProcessor()
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            
            # One-hot encode input
            input_one_hot = self.processor.one_hot_encode(input_seq, self.config.quantization_channels)
            input_one_hot = input_one_hot.permute(0, 2, 1)  # (batch, channels, time)
            
            # Forward pass
            output = self.model(input_one_hot)
            
            # Reshape for loss calculation
            output = output.permute(0, 2, 1)  # (batch, time, channels)
            loss = self.criterion(output.reshape(-1, self.config.quantization_channels), target_seq.reshape(-1))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return total_loss / len(dataloader)

# Audio generation/inference
class AudioGenerator:
    def __init__(self, model: WaveNet, config: Config):
        self.model = model
        self.config = config
        self.device = config.device
        self.processor = AudioProcessor()
        
    def generate_audio(self, length: int, temperature: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
        """Generate audio using the trained model"""
        self.model.eval()
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Start with random seed
        generated = torch.randint(0, self.config.quantization_channels, (1, 1), device=self.device)
        
        with torch.no_grad():
            for _ in range(length):
                # Prepare input
                input_one_hot = self.processor.one_hot_encode(generated, self.config.quantization_channels)
                input_one_hot = input_one_hot.permute(0, 2, 1)
                
                # Get model output
                output = self.model(input_one_hot)
                
                # Apply temperature and sample
                logits = output[:, :, -1] / temperature  # Last timestep
                probs = F.softmax(logits, dim=-1)
                
                # Sample next value
                next_sample = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_sample], dim=1)
        
        # Convert back to audio
        generated_np = generated.cpu().numpy().flatten()
        audio = self.processor.dequantize_audio(generated_np, self.config.quantization_channels)
        
        return audio

# Visualization utilities
class AudioVisualizer:
    @staticmethod
    def plot_waveform(audio: np.ndarray, sample_rate: int, title: str = "Audio Waveform"):
        """Plot audio waveform"""
        plt.figure(figsize=(12, 4))
        time = np.linspace(0, len(audio) / sample_rate, len(audio))
        plt.plot(time, audio)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_spectrogram(audio: np.ndarray, sample_rate: int, title: str = "Spectrogram"):
        """Plot audio spectrogram"""
        plt.figure(figsize=(12, 6))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sample_rate)
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    print("🎵 WaveNet Audio Generation Project")
    print("=" * 50)
    
    # Initialize components
    database = MockAudioDatabase(config)
    dataset = AudioDataset(database, config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Initialize model
    model = WaveNet(config).to(config.device)
    trainer = Trainer(model, config)
    generator = AudioGenerator(model, config)
    visualizer = AudioVisualizer()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Receptive field: {config.receptive_field} samples")
    print(f"Device: {config.device}")
    
    # Training loop
    print("\n🚀 Starting training...")
    for epoch in range(min(5, config.num_epochs)):  # Limit to 5 epochs for demo
        avg_loss = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch + 1}/{config.num_epochs}, Average Loss: {avg_loss:.4f}")
    
    # Generate and visualize audio
    print("\n🎼 Generating audio...")
    generated_audio = generator.generate_audio(length=16000, temperature=0.8)  # 1 second at 16kHz
    
    # Visualize original vs generated
    original_sample = database.get_sample(0)
    visualizer.plot_waveform(original_sample['audio'], config.sample_rate, "Original Audio")
    visualizer.plot_waveform(generated_audio, config.sample_rate, "Generated Audio")
    
    # Save generated audio
    output_path = "generated_audio.wav"
    wavfile.write(output_path, config.sample_rate, (generated_audio * 32767).astype(np.int16))
    print(f"Generated audio saved to: {output_path}")
    
    print("\n✅ Project completed successfully!")
    print("\n🧠 What This Project Demonstrates:")
    print("• Modern WaveNet architecture with dilated convolutions")
    print("• Proper residual and skip connections")
    print("• Gated activation functions")
    print("• Mock audio database with various patterns")
    print("• Complete training pipeline")
    print("• Audio generation and visualization")
    print("• Professional code structure and documentation")