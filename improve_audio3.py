import gradio as gr
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
# Prevent GUI crashes
import matplotlib

matplotlib.use('Agg')

# Constants for exact pixel mapping
F_MIN = 0
F_MAX = 22050
IMG_H = 1000  # Strictly forced height


def calculate_shelf_coeffs(f0, sr, gain_db, type='high'):
    """Stable High/Low Shelf coefficients (RBJ Cookbook)."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / 2 * np.sqrt(2)  # Q = 0.707
    cos_w0 = np.cos(w0)
    if type == 'high':
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


def rbj_peaking_sos(f0, sr, gain_db, Q=35.0):
    """RBJ Parametric Peaking EQ – perfect for surgical cuts (gain_db < 0)."""
    if abs(gain_db) < 0.01:
        return None  # bypass
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * Q)
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


def generate_calibrated_spec(input_file, selected_freqs=[]):
    if not input_file:
        return None
    y, sr = librosa.load(input_file, sr=None, mono=True, duration=60)
    # 1600x1000 px resolution
    fig = plt.figure(figsize=(16, 10), dpi=100)
    fig.set_facecolor('#0b0f19')
    # Leave space for y labels
    ax = fig.add_axes([0.1, 0.0, 0.9, 1.0])
    ax.set_facecolor('#0b0f19')
    # STFT and Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=4096, hop_length=4096)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis=None, y_axis='linear', ax=ax, cmap='magma', vmin=-75, vmax=0)
    # Lock limits precisely
    ax.set_ylim(F_MIN, F_MAX)
    # Detailed grid with submarks
    ax.yaxis.set_major_locator(MultipleLocator(2000))
    ax.yaxis.set_minor_locator(MultipleLocator(500))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='major', color='#ffffff', linestyle='-', alpha=0.25)
    ax.grid(True, which='minor', color='#ffffff', linestyle=':', alpha=0.1)
    # Set label and tick colors
    ax.set_ylabel('Frequency (Hz)', color='#ffffff')
    ax.tick_params(axis='y', colors='#ffffff', labelsize=8)
    # Selection lines
    for f in selected_freqs:
        ax.axhline(y=f, color='#00ffff', linewidth=2, alpha=0.8)
    fname = "surgical_spec.png"
    plt.savefig(fname, pad_inches=0, transparent=False)
    plt.close()
    return fname


def process_audio(input_file, notch_df, air_db, gain_db, notch_cut_db, notch_q):
    if not input_file:
        return None, None
    y, sr = librosa.load(input_file, sr=None, mono=False, dtype=np.float64)
    if y.ndim == 1:
        y = np.stack([y, y])

    # 1. 15Hz BRICKWALL CUT (8th Order)
    sos_hp = signal.butter(8, 15, 'hp', fs=sr, output='sos')
    y = signal.sosfilt(sos_hp, y, axis=-1)

    # 2. SURGICAL CUTS – now fully adjustable strength + wideness
    selected_freqs = []
    if notch_df is not None and len(notch_df) > 0:
        freq_list = notch_df.iloc[:, 0].tolist()
        sos_cuts = []
        for f in freq_list:
            if 15 < f < sr / 2 and notch_cut_db > 0:
                sos = rbj_peaking_sos(f, sr, gain_db=-notch_cut_db, Q=notch_q)
                if sos is not None:
                    sos_cuts.append(sos)
                    selected_freqs.append(f)
        if sos_cuts:
            y = signal.sosfilt(np.vstack(sos_cuts), y, axis=-1)

    # 3. AIR BOOST (Correct High Shelf)
    if air_db > 0:
        sos_shelf = calculate_shelf_coeffs(15000, sr, air_db, type='high')
        y = signal.sosfilt(np.atleast_2d(sos_shelf), y, axis=-1)

    # 4. GAIN & SOFT LIMIT
    y = np.tanh(y * (10 ** (gain_db / 20.0)))

    out_path = "surgical_master.flac"
    sf.write(out_path, y.T, sr, subtype='PCM_24')
    spec_out = generate_calibrated_spec(out_path, selected_freqs)
    return out_path, spec_out


# --- UI ---
with gr.Blocks(theme=gr.themes.Base(), css="body { background-color: #0b0f19; }") as demo:
    gr.Markdown("# 🎛️ AI Distortion filter - Mastering Tool")
    gr.Markdown("Click resonance lines → add frequencies → adjust **Cut Strength** & **Wideness (Q)** below.<br>"
                "15 Hz brickwall + fully parametric surgical cuts + Air + Makeup. 24-bit FLAC output.")

    active_freqs = gr.State([])

    with gr.Row():
        with gr.Column(scale=4):
            spec_display = gr.Image(label="Linear Spectrogram (0 Hz – 22 kHz) – click to pick frequencies",
                                    interactive=True)
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="1. Load Audio")

            with gr.Group():
                gr.Markdown("### 🎯 Selected Frequencies")
                last_clicked = gr.Number(label="Last Clicked (Hz)", precision=0)
                add_btn = gr.Button("➕ Add Frequency", variant="secondary")
                freq_table = gr.Dataframe(headers=["Target Hz"], datatype=["number"], col_count=(1, "fixed"),
                                          interactive=True)
                clear_btn = gr.Button("🗑️ Clear List", variant="stop")

            with gr.Group():
                gr.Markdown("### 🔪 Surgical Cut Controls")
                notch_cut_db = gr.Slider(0, 36, value=12, step=0.5, label="Cut Strength (dB reduction)")
                notch_q = gr.Slider(1, 100, value=35, step=0.5, label="Wideness (Q factor)",
                                    info="Higher Q = narrower cut (surgical). Lower Q = wider cut.")

            with gr.Group():
                gr.Markdown("### ✨ Final Polish")
                air_slider = gr.Slider(0, 24, value=12, label="Air Boost (dB)")
                gain_slider = gr.Slider(0, 12, value=3, label="Makeup Gain (dB)")
                master_btn = gr.Button("🚀 MASTER AUDIO", variant="primary")

    audio_output = gr.Audio(label="Final Result", type="filepath")
    output_spec = gr.Image(label="Mastered Spectrogram")


    def on_click(evt: gr.SelectData):
        """Precise pixel-to-frequency linear mapping."""
        y_pixel = evt.index[1]
        f_exact = F_MAX * (1 - y_pixel / 1000.0)
        return int(round(f_exact))


    def add_freq(f, freqs, file):
        if f not in freqs and f > 0:
            freqs.append(f)
            freqs.sort()
        img = generate_calibrated_spec(file, freqs)
        return freqs, [[x] for x in freqs], img


    def update_from_table(df, file):
        if df is None or len(df) == 0:
            return [], generate_calibrated_spec(file, [])
        try:
            import pandas as pd
            if isinstance(df, pd.DataFrame):
                freqs = df.iloc[:, 0].dropna().tolist()
            else:
                freqs = [row[0] for row in df if row and isinstance(row[0], (int, float))]
            freqs = sorted(set([int(f) for f in freqs if f > 0]))
            img = generate_calibrated_spec(file, freqs)
            return freqs, img
        except:
            return [], generate_calibrated_spec(file, [])


    def clear(file):
        return [], [], generate_calibrated_spec(file, [])


    # Wire everything up
    audio_input.change(lambda x: generate_calibrated_spec(x, []), [audio_input], [spec_display])
    spec_display.select(on_click, None, [last_clicked])
    add_btn.click(add_freq, [last_clicked, active_freqs, audio_input], [active_freqs, freq_table, spec_display])
    clear_btn.click(clear, [audio_input], [active_freqs, freq_table, spec_display])
    freq_table.change(update_from_table, [freq_table, audio_input], [active_freqs, spec_display])

    # NEW: pass the two new sliders
    master_btn.click(
        process_audio,
        [audio_input, freq_table, air_slider, gain_slider, notch_cut_db, notch_q],
        [audio_output, output_spec]
    )

if __name__ == "__main__":
    demo.launch()