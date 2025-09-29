# WaveNet Audio Generation Project

Implementation of WaveNet for audio generation using PyTorch, featuring a complete training pipeline, audio generation capabilities, and an interactive web interface.

## Features

- **Modern WaveNet Architecture**: Implements the full WaveNet model with dilated convolutions, residual connections, and skip connections
- **Mock Audio Database**: Synthetic audio samples including sine waves, chords, frequency sweeps, and noise patterns
- **Complete Training Pipeline**: Full training loop with validation and loss tracking
- **Audio Generation**: Generate new audio samples with controllable temperature and seed parameters
- **Interactive Web UI**: Streamlit-based interface for model interaction and audio visualization
- **Audio Visualization**: Waveform plots and spectrograms for analysis
- **Professional Code Structure**: Modular design with proper documentation and type hints

## Architecture

### WaveNet Model Components

1. **Dilated Convolutions**: Exponentially increasing dilation rates (1, 2, 4, 8, ...)
2. **Gated Activation**: Tanh and sigmoid gates for non-linear transformations
3. **Residual Connections**: Skip connections to prevent vanishing gradients
4. **Skip Connections**: Direct paths from each layer to the output
5. **Causal Convolutions**: Ensures autoregressive generation

### Model Specifications

- **Sample Rate**: 16 kHz
- **Quantization**: 256 levels (8-bit)
- **Receptive Field**: 1024 samples (64ms at 16kHz)
- **Residual Channels**: 64
- **Skip Channels**: 256
- **Dilation Depth**: 10 layers

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd 0137_WaveNet_for_audio_generation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Project

#### Command Line Interface
```bash
python 0137.py
```

#### Web Interface
```bash
streamlit run app.py
```

The web interface will open at `http://localhost:8501`

## 📁 Project Structure

```
0137_WaveNet_for_audio_generation/
├── 0137.py                 # Main script with complete implementation
├── wavenet_model.py       # Modular WaveNet implementation
├── app.py                 # Streamlit web interface
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── generated_audio.wav   # Output audio file (generated)
```

## Audio Generation

### Parameters

- **Temperature**: Controls randomness (0.1 = deterministic, 2.0 = very random)
- **Length**: Duration of generated audio in seconds
- **Seed**: Random seed for reproducible generation

### Usage Example

```python
from wavenet_model import Config, WaveNet, AudioGenerator

# Initialize components
config = Config()
model = WaveNet(config)
generator = AudioGenerator(model, config)

# Generate audio
audio = generator.generate_audio(
    length=16000,      # 1 second at 16kHz
    temperature=0.8,   # Moderate randomness
    seed=42           # Reproducible
)
```

## Web Interface Features

### Model Information
- Parameter count and model statistics
- Receptive field visualization
- Device information (CPU/GPU)

### Training Data
- View original audio samples
- Download training samples
- Audio pattern analysis

### Audio Generation
- Interactive parameter controls
- Real-time audio generation
- Waveform and spectrogram visualization
- Audio download functionality

### Training
- Configurable training epochs
- Real-time loss tracking
- Training progress visualization

## Technical Details

### WaveNet Architecture

The implementation follows the original WaveNet paper with these key components:

1. **Input Embedding**: Converts quantized audio to dense representations
2. **Dilated Convolutions**: Capture long-range dependencies efficiently
3. **Gated Activation**: `z = tanh(W_f * x) ⊙ σ(W_g * x)`
4. **Residual Connections**: `h = h + W_r * z`
5. **Skip Connections**: Direct paths to output layers
6. **Output Layers**: Generate probability distributions over quantized values

### Training Process

1. **Data Preparation**: Quantize audio to discrete values
2. **Sequence Creation**: Create input-target pairs for autoregressive training
3. **Forward Pass**: Compute logits for next sample prediction
4. **Loss Calculation**: Cross-entropy loss between predicted and actual samples
5. **Backpropagation**: Update model parameters

### Audio Generation

1. **Initialization**: Start with random seed sample
2. **Autoregressive Generation**: Predict next sample given previous samples
3. **Temperature Sampling**: Control randomness in generation
4. **Dequantization**: Convert discrete values back to continuous audio

## Performance Metrics

- **Model Parameters**: ~1.2M parameters
- **Training Time**: ~5 minutes for 5 epochs (CPU)
- **Generation Speed**: ~100 samples/second
- **Memory Usage**: ~500MB during training

## 🔧 Configuration

Modify the `Config` class in `wavenet_model.py` to adjust:

- Sample rate and duration
- Model architecture (channels, depth)
- Training parameters (batch size, learning rate)
- Quantization levels

## Use Cases

- **Music Generation**: Create melodies and harmonies
- **Sound Effects**: Generate synthetic audio effects
- **Speech Synthesis**: Foundation for text-to-speech systems
- **Audio Compression**: Learn efficient audio representations
- **Research**: Study autoregressive audio modeling

## Limitations

- **Training Data**: Currently uses synthetic patterns only
- **Model Size**: Limited by computational resources
- **Generation Length**: Longer sequences require more memory
- **Quality**: Synthetic training data limits realism

## Future Enhancements

- [ ] Real audio dataset integration
- [ ] Conditional generation (text-to-speech)
- [ ] Multi-scale WaveNet architecture
- [ ] Real-time audio generation
- [ ] Model compression and optimization
- [ ] Advanced audio preprocessing
- [ ] Transfer learning capabilities

## References

- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- DeepMind for the original WaveNet paper
- PyTorch team for the excellent deep learning framework
- Streamlit team for the web interface framework
- The open-source community for various audio processing libraries

 
# WaveNet-Audio-Generation-Project
