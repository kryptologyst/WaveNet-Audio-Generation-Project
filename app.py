import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import wavfile
import io
import base64
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Import our WaveNet implementation
from wavenet_model import (
    Config, WaveNet, MockAudioDatabase, AudioDataset, 
    Trainer, AudioGenerator, AudioVisualizer, AudioProcessor
)

# Page configuration
st.set_page_config(
    page_title="🎵 WaveNet Audio Generation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'database' not in st.session_state:
    st.session_state.database = None
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []

def initialize_components():
    """Initialize WaveNet components"""
    config = Config()
    database = MockAudioDatabase(config)
    model = WaveNet(config).to(config.device)
    generator = AudioGenerator(model, config)
    
    st.session_state.config = config
    st.session_state.database = database
    st.session_state.model = model
    st.session_state.generator = generator

def plot_waveform_streamlit(audio: np.ndarray, sample_rate: int, title: str):
    """Plot waveform using Streamlit"""
    fig, ax = plt.subplots(figsize=(12, 4))
    time = np.linspace(0, len(audio) / sample_rate, len(audio))
    ax.plot(time, audio)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    return fig

def plot_spectrogram_streamlit(audio: np.ndarray, sample_rate: int, title: str):
    """Plot spectrogram using Streamlit"""
    fig, ax = plt.subplots(figsize=(12, 6))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sample_rate, ax=ax)
    plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
    ax.set_title(title)
    return fig

def create_audio_download(audio: np.ndarray, sample_rate: int, filename: str):
    """Create downloadable audio file"""
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Create WAV file in memory
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio_int16)
    buffer.seek(0)
    
    # Encode as base64
    b64 = base64.b64encode(buffer.read()).decode()
    
    # Create download link
    href = f'<a href="data:audio/wav;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 WaveNet Audio Generation</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Initialize components if not done
    if st.session_state.model is None:
        with st.spinner("Initializing WaveNet..."):
            initialize_components()
        st.success("✅ WaveNet initialized successfully!")
    
    # Sidebar controls
    st.sidebar.subheader("🎚️ Generation Parameters")
    
    temperature = st.sidebar.slider(
        "Temperature", 
        min_value=0.1, 
        max_value=2.0, 
        value=0.8, 
        step=0.1,
        help="Higher temperature = more random, Lower temperature = more deterministic"
    )
    
    length_seconds = st.sidebar.slider(
        "Length (seconds)", 
        min_value=0.5, 
        max_value=5.0, 
        value=2.0, 
        step=0.5
    )
    
    seed = st.sidebar.number_input(
        "Random Seed", 
        min_value=0, 
        max_value=10000, 
        value=42,
        help="Set to 0 for random generation"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Model Information")
        
        config = st.session_state.config
        model = st.session_state.model
        
        # Model metrics
        param_count = sum(p.numel() for p in model.parameters())
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Model Statistics</h4>
            <p><strong>Parameters:</strong> {param_count:,}</p>
            <p><strong>Receptive Field:</strong> {config.receptive_field} samples</p>
            <p><strong>Sample Rate:</strong> {config.sample_rate} Hz</p>
            <p><strong>Quantization:</strong> {config.quantization_channels} levels</p>
            <p><strong>Device:</strong> {config.device}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Training data info
        st.subheader("🎼 Training Data")
        database = st.session_state.database
        
        st.write("**Available Audio Patterns:**")
        for i, sample in enumerate(database.audio_samples):
            st.write(f"• {sample['name']} ({sample['type']})")
        
        # Show original samples
        if st.button("🎵 Show Original Samples"):
            st.subheader("Original Training Samples")
            
            for i, sample in enumerate(database.audio_samples[:3]):  # Show first 3
                st.write(f"**{sample['name']}**")
                
                # Waveform
                fig = plot_waveform_streamlit(sample['audio'], config.sample_rate, f"{sample['name']} Waveform")
                st.pyplot(fig)
                
                # Download link
                download_link = create_audio_download(sample['audio'], config.sample_rate, f"original_{sample['name']}.wav")
                st.markdown(download_link, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🎹 Audio Generation")
        
        # Generation button
        if st.button("🎼 Generate New Audio", type="primary"):
            with st.spinner("Generating audio..."):
                generator = st.session_state.generator
                config = st.session_state.config
                
                # Generate audio
                length_samples = int(length_seconds * config.sample_rate)
                seed_val = None if seed == 0 else seed
                
                generated_audio = generator.generate_audio(
                    length=length_samples,
                    temperature=temperature,
                    seed=seed_val
                )
                
                st.session_state.generated_audio = generated_audio
        
        # Display generated audio
        if 'generated_audio' in st.session_state:
            st.subheader("🎵 Generated Audio")
            
            generated_audio = st.session_state.generated_audio
            config = st.session_state.config
            
            # Waveform plot
            fig = plot_waveform_streamlit(generated_audio, config.sample_rate, "Generated Audio Waveform")
            st.pyplot(fig)
            
            # Spectrogram
            fig = plot_spectrogram_streamlit(generated_audio, config.sample_rate, "Generated Audio Spectrogram")
            st.pyplot(fig)
            
            # Audio player
            st.audio(generated_audio, sample_rate=config.sample_rate)
            
            # Download link
            download_link = create_audio_download(generated_audio, config.sample_rate, "generated_audio.wav")
            st.markdown(download_link, unsafe_allow_html=True)
            
            # Audio statistics
            st.subheader("📈 Audio Statistics")
            col_stats1, col_stats2 = st.columns(2)
            
            with col_stats1:
                st.metric("RMS", f"{np.sqrt(np.mean(generated_audio**2)):.4f}")
                st.metric("Peak", f"{np.max(np.abs(generated_audio)):.4f}")
            
            with col_stats2:
                st.metric("Mean", f"{np.mean(generated_audio):.4f}")
                st.metric("Std Dev", f"{np.std(generated_audio):.4f}")
    
    # Training section
    st.subheader("🚀 Model Training")
    
    col_train1, col_train2 = st.columns([1, 2])
    
    with col_train1:
        epochs = st.number_input("Training Epochs", min_value=1, max_value=20, value=5)
        
        if st.button("🏋️ Train Model"):
            with st.spinner("Training model..."):
                config = st.session_state.config
                model = st.session_state.model
                database = st.session_state.database
                
                # Create dataset and dataloader
                dataset = AudioDataset(database, config)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
                
                # Initialize trainer
                trainer = Trainer(model, config)
                
                # Training loop
                progress_bar = st.progress(0)
                loss_history = []
                
                for epoch in range(epochs):
                    avg_loss = trainer.train_epoch(dataloader)
                    loss_history.append(avg_loss)
                    
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    
                    st.write(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
                
                st.session_state.training_history = loss_history
                st.success(f"✅ Training completed! Final loss: {loss_history[-1]:.4f}")
    
    with col_train2:
        if st.session_state.training_history:
            st.subheader("📊 Training Progress")
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(st.session_state.training_history)
            ax.set_title("Training Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True)
            st.pyplot(fig)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🎵 WaveNet Audio Generation Project | Built with PyTorch & Streamlit</p>
        <p>This project demonstrates modern WaveNet architecture for audio generation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
