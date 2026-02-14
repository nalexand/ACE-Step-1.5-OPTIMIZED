"""
MusicBox — Simple Music Generation Jukebox
One-click genre-based music generation with continuous playback.
All code in one file, uses ACE-Step pipeline via imports.
"""

import os
import sys
import time
import random
import threading
import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# ── Project path setup ──────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Clear proxy settings that may affect Gradio
for _pv in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    os.environ.pop(_pv, None)

import gradio as gr
import torch

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

# ── Genre Presets ────────────────────────────────────────────────────────────
GENRES: Dict[str, Dict[str, Any]] = {
    "Lo-Fi Hip Hop": dict(bpm=75, key="C minor", ts="4", dur=60, inst=True, lang="unknown",
                          caption_hint="lo-fi hip hop beat, warm vinyl crackle, mellow piano chords, jazzy samples, chill downtempo groove"),
    "Jazz": dict(bpm=120, key="Bb major", ts="4", dur=90, inst=False, lang="en",
                 caption_hint="smooth jazz, walking bass, saxophone melody, swing feel, soft brushed drums"),
    "Classical Piano": dict(bpm=90, key="D major", ts="3", dur=120, inst=True, lang="unknown",
                            caption_hint="classical piano solo, expressive dynamics, romantic era style, arpeggiated chords"),
    "EDM / House": dict(bpm=128, key="A minor", ts="4", dur=60, inst=True, lang="unknown",
                        caption_hint="electronic dance music, four on the floor beat, synth lead, deep bass, energetic build-up and drop"),
    "Trap": dict(bpm=140, key="F# minor", ts="4", dur=60, inst=False, lang="en",
                 caption_hint="trap beat, heavy 808 bass, hi-hat rolls, dark atmospheric synths, hard-hitting snares"),
    "Pop": dict(bpm=110, key="G major", ts="4", dur=60, inst=False, lang="en",
                caption_hint="catchy pop song, polished production, hooks, upbeat tempo, bright synths and guitars"),
    "R&B / Soul": dict(bpm=85, key="Eb major", ts="4", dur=60, inst=False, lang="en",
                       caption_hint="smooth R&B, soulful vocals, groove, lush harmonies, warm keys and bass"),
    "Rock": dict(bpm=130, key="E minor", ts="4", dur=60, inst=False, lang="en",
                 caption_hint="rock song, electric guitar riffs, driving drums, bass groove, powerful energy"),
    "Metal": dict(bpm=160, key="D minor", ts="4", dur=60, inst=False, lang="en",
                  caption_hint="heavy metal, distorted guitars, double bass drums, aggressive vocals, powerful riffs"),
    "Ambient": dict(bpm=70, key="C major", ts="4", dur=120, inst=True, lang="unknown",
                    caption_hint="ambient soundscape, ethereal pads, reverb, slow evolving textures, peaceful and meditative"),
    "Reggae": dict(bpm=80, key="G major", ts="4", dur=60, inst=False, lang="en",
                   caption_hint="reggae, offbeat guitar skank, deep bass, one drop rhythm, island vibes"),
    "Country": dict(bpm=110, key="A major", ts="4", dur=60, inst=False, lang="en",
                    caption_hint="country song, acoustic guitar, fiddle, steel guitar, warm storytelling vocals"),
    "Folk": dict(bpm=100, key="D major", ts="3", dur=60, inst=False, lang="en",
                 caption_hint="folk song, acoustic guitar, gentle vocals, organic instrumentation, warm and intimate"),
    "Blues": dict(bpm=85, key="E minor", ts="4", dur=60, inst=False, lang="en",
                  caption_hint="blues, 12-bar progression, bending guitar notes, soulful vocals, shuffle rhythm"),
    "Funk": dict(bpm=105, key="Bb minor", ts="4", dur=60, inst=False, lang="en",
                 caption_hint="funk, tight groovy bass line, rhythmic guitar, brass stabs, syncopated drums"),
    "Bossa Nova": dict(bpm=130, key="F major", ts="4", dur=60, inst=True, lang="unknown",
                       caption_hint="bossa nova, nylon string guitar, gentle percussion, smooth flowing melody, Brazilian jazz"),
    "Synthwave": dict(bpm=118, key="A minor", ts="4", dur=60, inst=True, lang="unknown",
                      caption_hint="synthwave, retro 80s synths, pulsating arpeggios, neon atmosphere, driving beat"),
    "Drum & Bass": dict(bpm=174, key="D minor", ts="4", dur=60, inst=True, lang="unknown",
                        caption_hint="drum and bass, breakbeat drums, heavy sub bass, rolling rhythm, high energy"),
    "Chillstep": dict(bpm=140, key="C minor", ts="4", dur=90, inst=True, lang="unknown",
                      caption_hint="chillstep, melodic dubstep, ethereal female vocals, lush pads, deep wobble bass"),
    "K-Pop": dict(bpm=120, key="Ab major", ts="4", dur=60, inst=False, lang="ko",
                  caption_hint="K-pop, catchy hooks, polished production, dynamic arrangement, dance pop beat"),
    "J-Pop": dict(bpm=130, key="E major", ts="4", dur=60, inst=False, lang="ja",
                  caption_hint="J-pop, bright melody, upbeat tempo, anime-style arrangement, energetic vocals"),
    "Latin Pop": dict(bpm=96, key="F major", ts="4", dur=60, inst=False, lang="es",
                      caption_hint="Latin pop, reggaeton beat, tropical instruments, danceable rhythm, romantic feel"),
    "Cinematic Epic": dict(bpm=90, key="D minor", ts="4", dur=120, inst=True, lang="unknown",
                           caption_hint="cinematic epic orchestral, full orchestra, dramatic strings, powerful brass, timpani, heroic theme"),
    "Acoustic": dict(bpm=100, key="C major", ts="4", dur=60, inst=False, lang="en",
                     caption_hint="acoustic song, fingerpicking guitar, warm vocals, minimal production, intimate and natural"),
    "Punk Rock": dict(bpm=170, key="E minor", ts="4", dur=45, inst=False, lang="en",
                      caption_hint="punk rock, fast power chords, aggressive energy, raw vocals, driving drums"),
}

GENRE_NAMES = list(GENRES.keys())

# ── State ────────────────────────────────────────────────────────────────────
STATE_IDLE = "idle"
STATE_LOADING = "loading"
STATE_GENERATING = "generating"
STATE_PLAYING = "playing"
STATE_STOPPING = "stopping"

class MusicBoxState:
    """Thread-safe global state for the music generation pipeline."""
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.state = STATE_IDLE
        self.status_text = "Ready — select a genre and click Play"
        self.timer_start: Optional[float] = None
        self.current_audio_path: Optional[str] = None
        self.song_count = 0
        self.song_history: list = []
        # Pipeline objects
        self.dit_handler: Optional[AceStepHandler] = None
        self.llm_handler: Optional[LLMHandler] = None
        self.models_loaded = False
        # Generation thread
        self.gen_thread: Optional[threading.Thread] = None
        # Next audio ready to play (pre-generated)
        self.next_audio_path: Optional[str] = None
        self.audio_changed = False  # Flag to signal new audio for UI

    def set_status(self, state: str, text: str):
        with self.lock:
            self.state = state
            self.status_text = text

    def get_status(self):
        with self.lock:
            return self.state, self.status_text

    def is_stopped(self):
        return self.stop_event.is_set()

S = MusicBoxState()

OUTPUT_DIR = os.path.join(_THIS_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Gemini API helper ────────────────────────────────────────────────────────
def gemini_generate_caption_and_lyrics(api_key: str, genre_name: str,
                                        user_description: str,
                                        genre_preset: dict) -> dict:
    """Call Gemini Flash to generate a music caption and lyrics."""
    try:
        from google import generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        is_instrumental = genre_preset.get("inst", False)
        prompt = (
            f"You are a professional music producer. Generate content for a {genre_name} song.\n"
            f"User wants: {user_description}\n"
            f"BPM: {genre_preset['bpm']}, Key: {genre_preset['key']}, "
            f"Time signature: {genre_preset['ts']}/4\n\n"
        )
        if is_instrumental:
            prompt += (
                "This is an INSTRUMENTAL track (no vocals).\n"
                "Generate ONLY a detailed music caption (2-3 sentences) describing the mood, "
                "instruments, style, and atmosphere.\n"
                "Format:\nCAPTION: <your caption>\nLYRICS: [Instrumental]"
            )
        else:
            prompt += (
                "Generate:\n"
                "1. A detailed music caption (2-3 sentences) describing the mood, instruments, style\n"
                "2. Song lyrics with structure tags like [Verse], [Chorus], [Bridge]\n\n"
                "Format:\nCAPTION: <your caption>\nLYRICS:\n<your lyrics>"
            )

        response = model.generate_content(prompt)
        text = response.text.strip()

        caption = ""
        lyrics = "[Instrumental]" if is_instrumental else ""

        if "CAPTION:" in text:
            parts = text.split("LYRICS:", 1)
            caption = parts[0].replace("CAPTION:", "").strip()
            if len(parts) > 1:
                lyrics = parts[1].strip()
        else:
            caption = text[:300]

        return {"caption": caption, "lyrics": lyrics, "success": True}
    except Exception as e:
        return {"caption": "", "lyrics": "", "success": False, "error": str(e)}


# ── Model Loading ────────────────────────────────────────────────────────────
def ensure_models_loaded(offload_cpu: bool, offload_dit: bool) -> str:
    """Lazy-load models on first use. Returns status message."""
    if S.models_loaded:
        return "Models ready"

    S.set_status(STATE_LOADING, "⏳ Loading DiT model...")

    project_root = _THIS_DIR
    checkpoint_dir = os.path.join(project_root, "checkpoints")

    # ── DiT handler ──
    S.dit_handler = AceStepHandler()
    available_models = S.dit_handler.get_available_acestep_v15_models()
    if not available_models:
        return "❌ No ACE-Step models found in checkpoints/"
    # Prefer turbo model
    config_path = "acestep-v15-turbo" if "acestep-v15-turbo" in available_models else available_models[0]

    use_flash = S.dit_handler.is_flash_attention_available()
    status_msg, ok = S.dit_handler.initialize_service(
        project_root=project_root,
        config_path=config_path,
        device="auto",
        use_flash_attention=use_flash,
        offload_to_cpu=offload_cpu,
        offload_dit_to_cpu=offload_dit,
    )
    if not ok:
        return f"❌ DiT init failed: {status_msg}"

    # ── LLM handler ──
    S.set_status(STATE_LOADING, "⏳ Loading LLM 1.7B...")
    S.llm_handler = LLMHandler()
    available_lm = S.llm_handler.get_available_5hz_lm_models()
    # Prefer 1.7B model
    lm_model = None
    for m in (available_lm or []):
        if "1.7B" in m:
            lm_model = m
            break
    if not lm_model and available_lm:
        lm_model = available_lm[0]

    if lm_model:
        lm_status, lm_ok = S.llm_handler.initialize(
            checkpoint_dir=checkpoint_dir,
            lm_model_path=lm_model,
            backend="pt",
            device="auto",
            offload_to_cpu=offload_cpu,
            dtype=S.dit_handler.dtype,
        )
        if not lm_ok:
            print(f"⚠ LLM init warning: {lm_status}")
    else:
        print("⚠ No LLM models found — will generate without LLM reasoning")

    S.models_loaded = True
    return "✅ Models loaded"


# ── Generation Logic ─────────────────────────────────────────────────────────
def generate_one_song(genre_name: str, user_desc: str,
                      gemini_key: str, use_gemini: bool,
                      offload_cpu: bool, offload_dit: bool) -> Optional[str]:
    """Generate a single song. Returns path to audio file or None on failure/stop."""

    # 1. Load models if needed
    load_msg = ensure_models_loaded(offload_cpu, offload_dit)
    if "❌" in load_msg:
        S.set_status(STATE_IDLE, load_msg)
        return None
    if S.is_stopped():
        return None

    genre = GENRES.get(genre_name, GENRES["Lo-Fi Hip Hop"])

    # 2. Caption & lyrics
    caption = genre["caption_hint"]
    lyrics = "[Instrumental]" if genre["inst"] else ""

    if user_desc and user_desc.strip():
        caption = f"{user_desc.strip()}, {caption}"

    # Try Gemini first
    if use_gemini and gemini_key and gemini_key.strip():
        S.set_status(STATE_GENERATING, "🤖 Gemini generating lyrics...")
        gem = gemini_generate_caption_and_lyrics(gemini_key.strip(), genre_name, user_desc or "", genre)
        if gem["success"] and gem["caption"]:
            caption = gem["caption"]
            if gem["lyrics"]:
                lyrics = gem["lyrics"]
        else:
            print(f"Gemini fallback: {gem.get('error', 'unknown')}")

    if S.is_stopped():
        return None

    # 3. Generate music
    S.song_count += 1
    song_num = S.song_count
    S.set_status(STATE_GENERATING, f"🎵 Generating song #{song_num}...")
    S.timer_start = time.time()

    params = GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics=lyrics,
        instrumental=genre["inst"],
        bpm=genre["bpm"],
        keyscale=genre["key"],
        timesignature=genre["ts"],
        duration=float(genre["dur"]),
        vocal_language=genre["lang"],
        inference_steps=8,
        seed=-1,
        thinking=True,
        lm_temperature=0.85,
        lm_top_p=0.9,
        use_cot_metas=True,
        use_cot_caption=True,
        use_cot_language=True,
        use_constrained_decoding=True,
    )

    config = GenerationConfig(
        batch_size=1,
        use_random_seed=True,
        audio_format="flac",
    )

    try:
        # Use autocast to handle dtype mismatches between model layers
        # ResidualFSQ quantizer produces float32 but Linear weights are bfloat16
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        autocast_dtype = S.dit_handler.dtype if S.dit_handler else torch.bfloat16
        with torch.amp.autocast(device_type=device_type, dtype=autocast_dtype, enabled=(device_type == "cuda")):
            result = generate_music(
                dit_handler=S.dit_handler,
                llm_handler=S.llm_handler,
                params=params,
                config=config,
                save_dir=OUTPUT_DIR,
            )
    except Exception as e:
        S.set_status(STATE_IDLE, f"❌ Generation error: {e}")
        print(f"Generation error: {e}")
        import traceback; traceback.print_exc()
        return None

    if not result.success or not result.audios:
        S.set_status(STATE_IDLE, f"❌ Generation failed: {result.error or result.status_message}")
        return None

    audio_path = result.audios[0].get("path", "")
    if not audio_path or not os.path.exists(audio_path):
        S.set_status(STATE_IDLE, "❌ No audio file produced")
        return None

    elapsed = time.time() - (S.timer_start or time.time())
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    song_info = f"Song #{song_num} — {genre_name} — {elapsed:.0f}s — {ts}"
    with S.lock:
        S.song_history.insert(0, song_info)
        if len(S.song_history) > 50:
            S.song_history = S.song_history[:50]

    return audio_path


def generation_loop(genre_name: str, user_desc: str,
                    gemini_key: str, use_gemini: bool,
                    offload_cpu: bool, offload_dit: bool):
    """Background thread: continuously generate and queue songs."""
    S.stop_event.clear()

    while not S.is_stopped():
        path = generate_one_song(genre_name, user_desc, gemini_key, use_gemini,
                                 offload_cpu, offload_dit)
        if S.is_stopped() or path is None:
            break

        # Store the generated audio
        with S.lock:
            if S.current_audio_path is None:
                # First song — set as current
                S.current_audio_path = path
            else:
                # Queue as next
                S.next_audio_path = path
            S.audio_changed = True
            S.state = STATE_PLAYING
            S.status_text = f"🔊 Playing song #{S.song_count}"

        # Wait for the song to finish or be stopped
        # We wait for ~duration + buffer, but also check for stop frequently
        # The UI will advance to the next song via the timer callback
        genre = GENRES.get(genre_name, GENRES["Lo-Fi Hip Hop"])
        wait_duration = genre["dur"] + 10  # extra buffer seconds
        waited = 0.0
        while waited < wait_duration and not S.is_stopped():
            time.sleep(1.0)
            waited += 1.0
            # If UI consumed the next path, we can generate another
            with S.lock:
                if S.next_audio_path is None and S.state == STATE_PLAYING:
                    break  # UI consumed next, generate another

        # If we have no next queued, loop will generate one
        # If stop was requested, break
        if S.is_stopped():
            break

        # Advance: promote next to current if it exists
        with S.lock:
            if S.next_audio_path:
                S.current_audio_path = S.next_audio_path
                S.next_audio_path = None
                S.audio_changed = True
                S.status_text = f"🔊 Playing song #{S.song_count}"

    S.set_status(STATE_IDLE, "⏹ Stopped — ready to play again")


# ── Gradio Callbacks ─────────────────────────────────────────────────────────
def on_play(genre_name, user_desc, gemini_key, use_gemini, offload_cpu, offload_dit):
    """Start continuous generation."""
    # If already running, ignore
    if S.gen_thread and S.gen_thread.is_alive():
        return "Already running — click Stop first"

    # Reset state
    S.stop_event.clear()
    with S.lock:
        S.current_audio_path = None
        S.next_audio_path = None
        S.audio_changed = False
        S.timer_start = time.time()
        S.state = STATE_GENERATING
        S.status_text = "🎵 Starting generation..."

    S.gen_thread = threading.Thread(
        target=generation_loop,
        args=(genre_name, user_desc, gemini_key, use_gemini, offload_cpu, offload_dit),
        daemon=True,
    )
    S.gen_thread.start()
    return "🎵 Generation started..."


def on_stop():
    """Stop generation."""
    S.stop_event.set()
    S.set_status(STATE_STOPPING, "⏳ Stopping after current operation...")
    return "Stopping..."


def on_next():
    """Skip to next song — forces generation of a new one."""
    with S.lock:
        if S.next_audio_path:
            S.current_audio_path = S.next_audio_path
            S.next_audio_path = None
            S.audio_changed = True
            return "⏭ Skipped to next"
    return "No next song ready yet"


def poll_status():
    """Timer callback — returns (status_html, timer_text, audio_update, history_text)."""
    state, text = S.get_status()

    # Timer
    if S.timer_start and state in (STATE_GENERATING, STATE_LOADING):
        elapsed = time.time() - S.timer_start
        mins, secs = divmod(int(elapsed), 60)
        timer_text = f"⏱ {mins:02d}:{secs:02d}"
    elif state == STATE_PLAYING:
        timer_text = "🔊 Playing"
    else:
        timer_text = ""

    # Status with visual indicator
    if state == STATE_GENERATING:
        status_html = f'<div style="padding:12px;border-radius:8px;background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px solid #0f3460;color:#e94560;font-size:16px;text-align:center;">{text}<br><span style="color:#94a3b8;font-size:13px;">{timer_text}</span></div>'
    elif state == STATE_LOADING:
        status_html = f'<div style="padding:12px;border-radius:8px;background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px solid #533483;color:#e6b800;font-size:16px;text-align:center;">{text}<br><span style="color:#94a3b8;font-size:13px;">{timer_text}</span></div>'
    elif state == STATE_PLAYING:
        status_html = f'<div style="padding:12px;border-radius:8px;background:linear-gradient(135deg,#0d1117,#161b22);border:1px solid #238636;color:#3fb950;font-size:16px;text-align:center;">{text}</div>'
    else:
        status_html = f'<div style="padding:12px;border-radius:8px;background:linear-gradient(135deg,#0d1117,#161b22);border:1px solid #30363d;color:#8b949e;font-size:16px;text-align:center;">{text}</div>'

    # Audio update
    audio_update = gr.update()
    with S.lock:
        if S.audio_changed and S.current_audio_path:
            audio_update = gr.update(value=S.current_audio_path, autoplay=True)
            S.audio_changed = False

    # History
    history_text = "\n".join(S.song_history[:20]) if S.song_history else "No songs generated yet"

    return status_html, audio_update, history_text


def on_audio_end():
    """Called when audio finishes playing — advance to next song if available."""
    with S.lock:
        if S.next_audio_path:
            S.current_audio_path = S.next_audio_path
            S.next_audio_path = None
            S.audio_changed = True
            S.status_text = f"🔊 Playing song #{S.song_count}"
            return gr.update(value=S.current_audio_path, autoplay=True)
    return gr.update()


# ── CSS ──────────────────────────────────────────────────────────────────────
CSS = """
/* Dark theme overrides */
.gradio-container {
    max-width: 720px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
#title-row {
    text-align: center;
    padding: 8px 0 4px;
}
#title-row h1 {
    background: linear-gradient(135deg, #e94560, #533483, #0f3460);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2em;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
#title-row p {
    color: #8b949e;
    font-size: 0.95em;
    margin: 2px 0 0;
}
.play-btn {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    border: none !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    min-height: 52px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(35, 134, 54, 0.3) !important;
}
.play-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(35, 134, 54, 0.5) !important;
}
.stop-btn {
    background: linear-gradient(135deg, #da3633, #f85149) !important;
    border: none !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    min-height: 52px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(218, 54, 51, 0.3) !important;
}
.stop-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(218, 54, 51, 0.5) !important;
}
.skip-btn {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    min-height: 52px !important;
}
footer { display: none !important; }
"""

# ── Gradio UI ────────────────────────────────────────────────────────────────
def create_ui():
    with gr.Blocks(
        title="🎵 MusicBox",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="rose",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
    ) as demo:

        # ── Title ──
        gr.HTML(
            '<div id="title-row">'
            '<h1>🎵 MusicBox</h1>'
            '<p>Select a genre • Describe your vibe • Hit Play</p>'
            '</div>'
        )

        # ── Main Controls ──
        with gr.Column():
            genre_dd = gr.Dropdown(
                choices=GENRE_NAMES,
                value="Lo-Fi Hip Hop",
                label="🎸 Genre",
                interactive=True,
            )
            user_desc = gr.Textbox(
                label="✍️ Describe your music",
                placeholder="e.g. peaceful morning vibes, rainy day mood, energetic workout...",
                lines=2,
                max_lines=4,
            )

            with gr.Row():
                play_btn = gr.Button("▶  Play", elem_classes="play-btn", scale=2)
                stop_btn = gr.Button("⏹  Stop", elem_classes="stop-btn", scale=1)
                skip_btn = gr.Button("⏭  Next", elem_classes="skip-btn", scale=1)

        # ── Now Playing ──
        status_html = gr.HTML(
            '<div style="padding:12px;border-radius:8px;background:linear-gradient(135deg,#0d1117,#161b22);border:1px solid #30363d;color:#8b949e;font-size:16px;text-align:center;">Ready — select a genre and click Play</div>'
        )
        audio_player = gr.Audio(
            label="🔊 Now Playing",
            interactive=False,
            autoplay=True,
            type="filepath",
        )

        # ── History ──
        with gr.Accordion("📜 Song History", open=False):
            history_box = gr.Textbox(
                label="Generated Songs",
                lines=8,
                interactive=False,
                value="No songs generated yet",
            )

        # ── Settings (collapsed) ──
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                offload_cpu = gr.Checkbox(label="CPU Offload", value=True,
                                          info="Offload models to CPU when not in use (saves VRAM)")
                offload_dit = gr.Checkbox(label="DiT CPU Offload", value=False,
                                          info="Also offload DiT model (for very low VRAM)")
            with gr.Row():
                gemini_key = gr.Textbox(
                    label="Gemini API Key (optional)",
                    placeholder="AIza...",
                    type="password",
                    scale=3,
                )
                use_gemini = gr.Checkbox(label="Use Gemini for lyrics", value=False, scale=1)

        # ── Timer for polling ──
        timer = gr.Timer(value=2)

        # ── Events ──
        play_btn.click(
            fn=on_play,
            inputs=[genre_dd, user_desc, gemini_key, use_gemini, offload_cpu, offload_dit],
            outputs=[status_html],
        )
        stop_btn.click(fn=on_stop, outputs=[status_html])
        skip_btn.click(fn=on_next, outputs=[status_html])

        timer.tick(
            fn=poll_status,
            outputs=[status_html, audio_player, history_box],
        )

        # Audio end event — advance to next song
        audio_player.stop(
            fn=on_audio_end,
            outputs=[audio_player],
        )

    return demo


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("🎵 MusicBox — Simple Music Generation Jukebox")
    print("=" * 60)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Genres available: {len(GENRES)}")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")
    else:
        print("  GPU: None detected (CPU mode)")
    print("=" * 60 + "\n")

    demo = create_ui()
    demo.queue(max_size=10)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        show_error=True,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
