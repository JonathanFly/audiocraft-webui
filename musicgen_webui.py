"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from tempfile import NamedTemporaryFile
import argparse
import torch
import gradio as gr
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


import torch
import torchaudio
import gc


import numpy as np
import scipy.io.wavfile as wavfile


from gradio_util import util

OFFLOAD_CPU = True
where_am_i = os.getcwd()


base_theme = gr.themes.Base()
default_theme = gr.themes.Default()
monochrome_theme = gr.themes.Monochrome()
soft_theme = gr.themes.Soft()
glass_theme = gr.themes.Glass()

allowed_paths_list = [os.path.join(where_am_i, "musicgen_samples")]

os.makedirs(allowed_paths_list[0], exist_ok=True)

MODEL = None
IS_SHARED_SPACE = "musicgen/MusicGen" in os.environ.get("SPACE_ID", "")


def load_model(version):
    print("Loading model", version)

    model = None
    try:
        model = MusicGen.get_pretrained(version)
    except Exception as e:
        print(f"Model Error: {e}")
        clean_models()
        return None
    return model


import json


def gradio_generate(
    model,
    text,
    melody,
    duration,
    continuation_overlap,
    topk,
    topp,
    temperature,
    cfg_coef,
    output_directory,
    continuation_audio,
):
    # Capture function parameters
    parameters = locals().copy()

    # Convert parameters to a JSON string
    parameters_json = json.dumps(parameters, default=str)

    sample_rate, output = predict(
        model,
        text,
        melody,
        duration,
        continuation_overlap,
        topk,
        topp,
        temperature,
        cfg_coef,
        output_directory,
        continuation_audio,
    )

    # Write audio file and return the filename without extension
    output_filename = util.write_audio_file(output_directory, output, sample_rate, text)

    # Check if the output_filename is not None before proceeding
    if output_filename is not None:
        # Write the parameters to a JSON file with the same name as the audio file
        try:
            with open(output_filename + ".json", "w") as f:
                print(f"parameters_json: {parameters_json}")
                # f.write(parameters_json)
            print(f"Parameters saved to {output_filename + '.json'}")
        except Exception as e:
            print(f"Error occurred while writing parameters: {str(e)}")

    return sample_rate, output


def grab_audio_segment(
    audio, last_n_seconds, seconds_from_back=None, sample_rate=32000
):
    total_seconds = audio.shape[-1] / sample_rate
    if seconds_from_back is not None and seconds_from_back > total_seconds:
        print(
            f"Warning: seconds_from_back ({seconds_from_back}s) is greater than the total length of the audio ({total_seconds}s). Defaulting to grabbing from the end of the audio."
        )
        seconds_from_back = None

    n_samples = int(last_n_seconds * sample_rate)
    if seconds_from_back is not None:
        back_samples = int(seconds_from_back * sample_rate)
        if n_samples > back_samples:
            print(
                f"Warning: last_n_seconds ({last_n_seconds}s) is greater than seconds_from_back ({seconds_from_back}s). Defaulting to grabbing from the end of the audio."
            )
            seconds_from_back = None
        else:
            audio_segment = audio[..., back_samples - n_samples : back_samples]
    if seconds_from_back is None:
        audio_segment = audio[..., -n_samples:]

    original_size = audio.size()
    print(f"Original size: {original_size}")
    modified_size = audio_segment.size()
    print(f"Modified size: {modified_size}")

    return audio_segment


def _clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# is this overkill? I copied this from Bark because switching models was trouble.
def clean_models():
    global MODEL

    del MODEL
    MODEL = None

    _clear_cuda_cache()
    gc.collect()


def predict(
    model,
    text,
    melody,
    duration,
    continuation_overlap,
    topk,
    topp,
    temperature,
    cfg_coef,
    output_directory,
    continuation_audio,
):
    global MODEL
    topk = int(topk)
    if MODEL is None or MODEL.name != model:
        clean_models()
        MODEL = load_model(model)

    if duration > MODEL.lm.cfg.dataset.segment_duration:
        duration = MODEL.lm.cfg.dataset.segment_duration
        print("MusicGen currently supports durations of up to 30 seconds.")

    MODEL.set_generation_params(
        use_sampling=True,
        top_k=topk,
        top_p=topp,
        temperature=temperature,
        cfg_coef=cfg_coef,
        duration=duration,
    )

    if melody is not None:
        print(f"Melody: {melody}")
        melody_waveform, sr = torchaudio.load(melody)

        if melody_waveform.dim() == 2:
            melody_waveform = melody_waveform[None]

        melody_wavform = grab_audio_segment(melody_wavform, duration)

        output = MODEL.generate_with_chroma(
            descriptions=[text],
            melody_wavs=melody_wavform,
            melody_sample_rate=sr,
            progress=True,
        )

    # Music Continuation
    elif continuation_audio is not None and isinstance(
        continuation_audio, (str, bytes, os.PathLike)
    ):
        print(f"continuation_audio: {continuation_audio}")
        continuation_wavform, sr = torchaudio.load(continuation_audio)
        full_continuation_wavform = (
            continuation_wavform.clone()
        )  # Save the full wavform
        # print(continuation_wavform.shape)
        if continuation_wavform.dim() == 2:
            continuation_wavform = continuation_wavform[None]

        # continuation_wavform = melody[..., -int(sr * continuation_overlap):]
        continuation_wavform = grab_audio_segment(
            continuation_wavform, continuation_overlap
        )

        output = MODEL.generate_continuation(
            prompt=continuation_wavform,
            prompt_sample_rate=sr,
            descriptions=[text],
            progress=True,
        )

        full_continuation_wavform = full_continuation_wavform.float()

        output = torch.cat(
            [full_continuation_wavform, output.detach().cpu().squeeze(0)], dim=-1
        )

    # Unconditional Generation
    else:
        print(f"Text: {text}")
        output = MODEL.generate(descriptions=[text], progress=True)

    output = output.detach().cpu().numpy()
    return MODEL.sample_rate, output


MusicGen_CSS = """

"""


with gr.Blocks(theme=default_theme, allowed_paths=allowed_paths_list) as demo:
    with gr.Column(scale=2):
        gr.Markdown(
            """
            # MusicGen
            This is your private demo for [MusicGen](https://github.com/facebookresearch/audiocraft), a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284)
            """
        )
    with gr.Column(scale=1):
        gr.Markdown(
            """
            
            """
        )
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                text = gr.TextArea(
                    label="Text Input - Describe the music.",
                    info='"Text Conditioning"',
                    interactive=True,
                )
                melody = gr.Audio(
                    source="upload",
                    type="filepath",
                    label="Audio Input - Melody Guided Generation",
                    info='"Melody Conditioning"',
                    interactive=True,
                )
                continuation_audio = gr.Audio(
                    source="upload",
                    type="filepath",
                    label="Music Continuation",
                    info="",
                    interactive=True,
                )
            with gr.Row():
                submit = gr.Button("Generate Audio")
            with gr.Row():
                model = gr.Radio(
                    ["medium", "small", "large", "melody"],
                    label="Model",
                    value="medium",
                    interactive=True,
                )
            with gr.Row():
                duration = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=30,
                    label="Audio Clip Duration",
                    interactive=True,
                    info="How long of a clip to generate. If you are running out of GPU ram and you don't want to use a smaller model, lower this.",
                )
                continuation_overlap = gr.Slider(
                    minimum=0,
                    maximum=30,
                    value=20,
                    label="Audio Clip Continuation Overlap. (Continuation only)",
                    info="MusicGen uses this many seconds of the original song, then uses the rest of the clip to continue it.",
                )
            with gr.Row(variant="panel"):
                with gr.Column(scale=3):
                    with gr.Row():
                        cfg_coef = gr.Slider(
                            label="Classifier Free Guidance",
                            info="This is probably the knob you want to tweak the most. Lots of possibilities.",
                            value=3.0,
                            interactive=True,
                            minimum=-5.0,
                            maximum=10.0,
                        )

                    with gr.Row():
                        topk = gr.Slider(
                            label="topk",
                            value=250,
                            minimum=0,
                            maximum=1000,
                            step=1,
                        )
                        topp = gr.Slider(
                            label="topp",
                            value=0.0,
                            minimum=0.0,
                            maximum=1.0,
                        )

                        temperature = gr.Slider(
                            label="temperature: ",
                            info="",
                            minimum=0.000,
                            maximum=3.0,
                            value=1.00,
                            interactive=True,
                        )
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        ***Classifier Free Guidance*** is how strongly the audio tries to stock to your instructions or the melody. 3.0 is what they used for melody in conditioning in the training, but may try lower values for pure text I think. 

                        ***Negative Guidance*** is very interesting and and does sometimes do what you would hope, give you the 'opposite' music of what you described. Try lightly negative values, with the large model. (Don't get too excited, but it is pretty neat.)
                        """
                    )

        with gr.Column(scale=1):
            output_directory = gr.Text(
                label=f"Output Directory in {where_am_i}/",
                interactive=True,
                value="musicgen_samples",
            )
            output = gr.Audio(label="Generated Music", type="numpy")
    submit.click(
        gradio_generate,
        inputs=[
            model,
            text,
            melody,
            duration,
            continuation_overlap,
            topk,
            topp,
            temperature,
            cfg_coef,
            output_directory,
            continuation_audio,
        ],
        outputs=[output],
    )
    gr.Examples(
        fn=predict,
        examples=[
            [
                "A jovial pirate shanty with hearty group vocals, buoyant banjo plucking, and rhythmic synth pads in the background, providing an irresistible invitation to join the rollicking camaraderie.",
                None,
                "large",
            ],
            [
                "An 80s driving pop song with heavy drums and synth pads in the background",
                "./assets/bach.mp3",
                "melody",
            ],
            [
                "A cheerful country song with acoustic guitars",
                "./assets/bolero_ravel.mp3",
                "melody",
            ],
            ["90s rock song with electric guitar and heavy drums", None, "medium"],
            [
                "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions",
                "./assets/bach.mp3",
                "melody",
            ],
            [
                "lofi slow bpm electro chill with organic samples",
                None,
                "medium",
            ],
        ],
        inputs=[text, melody, model],
        outputs=[output],
    )


import argparse

parser = argparse.ArgumentParser(description="Gradio app command line options.")
parser.add_argument("--share", action="store_true", help="Enable share setting.")
parser.add_argument("--user", type=str, help="User for authentication.")
parser.add_argument("--password", type=str, help="Password for authentication.")
parser.add_argument("--listen", action="store_true", help="Server name setting.")
parser.add_argument("--server_port", type=int, default=7860, help="Port setting.")
parser.add_argument(
    "--no-autolaunch",
    action="store_false",
    default=False,
    help="Disable automatic opening of the app in browser.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Enable detailed error messages and extra outputs.",
)


args = parser.parse_args()
auth = None


share = args.share


if args.user and args.password:
    auth = (args.user, args.password)

if args.share and auth is None:
    print("You may want to set a password, you are sharing this Gradio publicly.")


server_name = "0.0.0.0" if args.listen else "127.0.0.1"

if args.debug:
    print(util.gpu_status_report())

print(f"Model Download Dirs:")


torch_home = os.path.expanduser(
    os.getenv(
        "TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch")
    )
)

print(torch_home)

if args.debug:
    print(f"HF_HOME: {os.getenv('HF_HOME')}")
    print(f"XDG_CACHE_HOME: {os.getenv('XDG_CACHE_HOME')}")

print(f"\nYou should see Music Gen in your web browser now.")
print(f"If not go the the website you see below as 'Running on local URL:'")
print(f"python app.py --help for specific Gradio options like turning on sharing.\n\n")
# demo.queue(concurrency_count=2, max_size=2)
demo.queue()

do_not_launch = not args.no_autolaunch

do_not_launch = True

demo.launch(
    share=args.share,
    auth=auth,
    server_name=server_name,
    server_port=args.server_port,
    inbrowser=do_not_launch,
    debug=args.debug,
)

# Only auto launch one time.
do_not_launch = True


demo.launch()
