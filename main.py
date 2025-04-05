import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, AnimateDiffPipeline, DDIMScheduler
from moviepy.editor import VideoClip, AudioFileClip, CompositeAudioClip
import cv2
import numpy as np
import face_recognition
import gradio as gr
from gfpgan import GFPGANer
import librosa
from transformers import BarkModel
import soundfile as sf
import random
import os
from realesrgan import RealESRGANer
from pathlib import Path
import subprocess
import shutil

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FPS = 24
TARGET_DURATIONS = {"18 min": 18 * 60, "23 min": 23 * 60}

# --- Load Models ---
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
).to(DEVICE)

animate_pipe = AnimateDiffPipeline.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16
).to(DEVICE)

bark_model = BarkModel.from_pretrained("suno/bark-small").to(DEVICE)

face_enhancer = GFPGANer(
    model_path="GFPGANv1.4.pth", upscale=2, arch="clean", channel_multiplier=2, bg_upsampler=None
)

upscaler = RealESRGANer(
    model_path="RealESRGAN_x4plus.pth", scale=4, tile=0, tile_pad=10, pre_pad=0, half=True
)

# --- Prompt Enhancement ---
def enhance_prompt(scene, character_names):
    enhancements = {
        "chaos": ", panicked crowd, screaming, debris flying, dark smoke, flickering lights",
        "battle": ", epic clash, sparks, dynamic poses, intense lighting, dust clouds",
        "busy highway": ", roaring traffic, honking cars, trucks, motorcycles, exhaust fumes",
        "busy crosswalk": ", people walking, texting, carrying bags, street vendors, chatter",
        "calm": ", gentle breeze, soft lighting, serene expressions, flowing hair",
        "rain": ", wet streets, reflections, umbrellas, raindrops, moody atmosphere"
    }
    
    base_prompt = scene
    for key, value in enhancements.items():
        if key in scene.lower():
            base_prompt += value
    
    angles = ["low-angle shot", "high-angle shot", "wide shot", "close-up"]
    details = ["subtle wind effects", "dramatic shadows", "vibrant colors"]
    base_prompt += f", {random.choice(angles)}, {random.choice(details)}"
    
    for char in character_names:
        if char in scene:
            base_prompt += f", {char} with detailed expression"
    
    return base_prompt

# --- Voice Generation ---
def generate_voice(script, voice_preset="v2/en_speaker_6"):
    inputs = bark_model.prepare_inputs(script)
    audio = bark_model.generate(**inputs, voice_preset=voice_preset)
    return audio.cpu().numpy()

# --- Auto Lip-Sync with Wav2Lip ---
def lip_sync(frames, audio_path):
    # Save frames as a temporary video
    temp_video_path = "temp_input.mp4"
    temp_output_path = "temp_synced.mp4"
    frame_height, frame_width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video_path, fourcc, FPS, (frame_width, frame_height))
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    
    # Run Wav2Lip inference
    wav2lip_path = Path("Wav2Lip/inference.py")
    checkpoint_path = Path("Wav2Lip/checkpoints/wav2lip_gan.pth")
    if not wav2lip_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError("Wav2Lip files not found. Ensure Wav2Lip is cloned and model is downloaded.")
    
    command = [
        "python", str(wav2lip_path),
        "--checkpoint_path", str(checkpoint_path),
        "--face", temp_video_path,
        "--audio", audio_path,
        "--outfile", temp_output_path
    ]
    subprocess.run(command, check=True)
    
    # Read synced video back into frames
    cap = cv2.VideoCapture(temp_output_path)
    synced_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        synced_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    # Clean up temp files
    os.remove(temp_video_path)
    os.remove(temp_output_path)
    
    return synced_frames

# --- Crowd SFX ---
def add_sfx(audio, scene_type):
    sfx_map = {
        "chaos": "scream_sfx.wav",
        "battle": "battle_sfx.wav",
        "busy highway": "traffic_sfx.wav",
        "busy crosswalk": "crowd_sfx.wav"
    }
    for key, sfx_file in sfx_map.items():
        if key in scene_type.lower() and os.path.exists(sfx_file):
            sfx, _ = librosa.load(sfx_file, sr=22050)
            audio = np.concatenate([audio, sfx[:len(audio)]])
    return audio

# --- Generate Battle Scenes ---
def generate_battle_scene(prompt, duration_sec=10, fps=FPS):
    frames = []
    for i in range(duration_sec * fps):
        frame = pipe(prompt + f", frame {i}", num_inference_steps=30).images[0]
        frame_np = np.array(frame)
        enhanced_frame = upscaler.enhance(frame_np)[0]
        frames.append(enhanced_frame)
    return animate_pipe(frames).frames

# --- Generate Normal Scenes ---
def generate_normal_scene(prompt, duration_sec=10, fps=FPS):
    frames = []
    for i in range(duration_sec * fps):
        frame = pipe(prompt + f", frame {i}", num_inference_steps=20).images[0]
        frame_np = np.array(frame)
        enhanced_frame = upscaler.enhance(frame_np)[0]
        frames.append(enhanced_frame)
    return frames

# --- Full Episode Generator ---
def generate_anime_episode(script, character_images, character_names, target_duration="23 min"):
    scenes = script.split("\n---\n")
    final_frames = []
    final_audio = []
    total_duration_sec = 0
    
    char_map = {name: img for name, img in zip(character_names.split(","), character_images)}
    
    for scene in scenes:
        enhanced_prompt = enhance_prompt(scene, char_map.keys())
        duration_sec = random.randint(20, 60)
        
        if "battle" in scene.lower() or "chaos" in scene.lower():
            frames = generate_battle_scene(enhanced_prompt, duration_sec)
        else:
            frames = generate_normal_scene(enhanced_prompt, duration_sec)
        
        # Lip-sync if dialogue exists
        if ":" in scene:
            char_name, dialogue = scene.split(":", 1)
            if char_name.strip() in char_map:
                audio = generate_voice(dialogue.strip())
                audio_path = "temp_audio.wav"
                sf.write(audio_path, audio, 22050)
                frames = lip_sync(frames, audio_path)
                final_audio.append(audio)
                os.remove(audio_path)
        
        # Add SFX
        if any(k in scene.lower() for k in ["chaos", "battle", "busy highway", "busy crosswalk"]):
            audio_with_sfx = add_sfx(audio if ":" in scene else np.zeros(22050 * duration_sec), scene)
            final_audio.append(audio_with_sfx)
        
        final_frames.extend(frames)
        total_duration_sec += duration_sec
    
    # Pad to target duration
    target_sec = TARGET_DURATIONS[target_duration]
    if total_duration_sec < target_sec:
        padding_sec = target_sec - total_duration_sec
        filler_prompt = f"filler scene, calm landscape, {random.choice(list(char_map.keys()))} reflecting, soft music"
        filler_frames = generate_normal_scene(filler_prompt, padding_sec)
        final_frames.extend(filler_frames)
        total_duration_sec = target_sec
    
    # Combine audio
    if final_audio:
        final_audio_np = np.concatenate(final_audio)
        sf.write("episode_audio.wav", final_audio_np, 22050)
        audio_clip = AudioFileClip("episode_audio.wav").subclip(0, total_duration_sec)
    else:
        audio_clip = None
    
    # Render video
    video_path = "anime_episode.mp4"
    clip = VideoClip(lambda t: final_frames[int(t * FPS)], duration=total_duration_sec)
    if audio_clip:
        clip = clip.set_audio(audio_clip)
    clip.write_videofile(video_path, fps=FPS, codec="libx264", audio_codec="aac")
    
    return video_path

# --- Modern Gradio UI ---
with gr.Blocks(theme="soft", title="🎬 AI Anime Revolution Studio") as app:
    gr.Markdown(
        """
        # 🌟 AI Anime Revolution Studio
        Create cinematic anime episodes with cutting-edge AI—all for free!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tab("📝 Script"):
                script_input = gr.TextArea(label="Episode Script (Separate scenes with '---')", lines=10)
            
            with gr.Tab("🎨 Characters"):
                char_images = gr.Gallery(label="Upload Character Images", preview=True)
                char_names = gr.Textbox(label="Character Names (comma-separated, e.g., 'Vihaan, Aiko')")
            
            with gr.Tab("⚙️ Settings"):
                duration_choice = gr.Dropdown(
                    choices=["18 min", "23 min"], label="Episode Length", value="23 min"
                )
                gr.Markdown("### Advanced Options (Coming Soon)")
        
        with gr.Column(scale=3):
            with gr.Tab("⚡ Generate"):
                generate_btn = gr.Button("Generate Episode", variant="primary")
                output_video = gr.Video(label="Your Anime Episode", interactive=False)
                progress = gr.Textbox(label="Progress", interactive=False)
    
    generate_btn.click(
        fn=generate_anime_episode,
        inputs=[script_input, char_images, char_names, duration_choice],
        outputs=output_video,
        _js="() => {document.querySelector('.progress').value = 'Generating...';}"
    ).then(
        lambda: "Done!", outputs=progress
    )

app.launch(share=True)
