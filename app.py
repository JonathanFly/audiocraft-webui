# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Updated to account for UI changes from https://github.com/rkfg/audiocraft/blob/long/app.py
# also released under the MIT license.

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import subprocess as sp
from tempfile import NamedTemporaryFile
import time
import warnings

import json
import torch
import gradio as gr

import pathlib
from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen

from utils import set_seed, parse_or_set_seed, generate_random_seed
import torchaudio
import datetime
import shutil

import json
import ast
import operator as op


css = ""
MODEL = None  # Last used model
IS_BATCHED = "facebook/MusicGen" in os.environ.get("SPACE_ID", "")
if IS_BATCHED:
    from assets.share_btn import community_icon_html, loading_icon_html, share_js, css

MAX_BATCH_SIZE = 12
BATCHED_DURATION = 15
INTERRUPTING = False
# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call


where_am_i = os.getcwd()
output_dir = os.path.join(where_am_i, "musicgen_samples")


IN_GRADIO_SPACE = True


def _call_nostderr(*args, **kwargs):
    # Avoid ffmpeg vomitting on the logs.
    kwargs["stderr"] = sp.DEVNULL
    kwargs["stdout"] = sp.DEVNULL
    _old_call(*args, **kwargs)


sp.call = _call_nostderr
# Preallocating the pool of processes.
pool = ProcessPoolExecutor(4)
pool.__enter__()


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out


def load_model(version="melody"):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        MODEL = MusicGen.get_pretrained(version)


GENERATION_SEQUENCES = None


def grab_audio_segment(
    audio,
    last_n_seconds,
    sample_to_this_many_seconds_instead_of_end=None,
    sample_rate=32000,
):
    total_seconds = audio.shape[-1] / sample_rate

    n_samples = int(last_n_seconds * sample_rate)

    print(
        f"Grabbing {last_n_seconds}s ({n_samples} samples) from the audio of length {total_seconds}."
    )

    if sample_to_this_many_seconds_instead_of_end is not None:
        print(
            f"sample_to_this_many_seconds_instead_of_end: {sample_to_this_many_seconds_instead_of_end}"
        )

        front_samples = int(sample_to_this_many_seconds_instead_of_end * sample_rate)
        if n_samples > front_samples:
            print(
                f"Warning: last_n_seconds ({last_n_seconds}s) is greater than sample_to_this_many_seconds_instead_of_end ({sample_to_this_many_seconds_instead_of_end}s). Defaulting to grabbing from the end of the audio."
            )
            sample_to_this_many_seconds_instead_of_end = None
        else:
            print(f"Grabbing from {front_samples - n_samples} to {front_samples}")
            audio_segment = audio[..., front_samples - n_samples : front_samples]

    else:
        audio_segment = audio[..., -n_samples:]

    print(f"Original size: {total_seconds}")
    modified_size = audio_segment.shape[-1] / sample_rate
    print(f"Modified size: {modified_size}")

    return audio_segment


def _do_predictions(
    model_name,
    texts,
    melodies,
    duration,
    continuation_overlap,
    continuation_audio,
    seed,
    progress=False,
    **gen_kwargs,
):
    # I want to able to reference past audio uploads in a Space, via JSON so I'm going to try to switch to filename paths, cause I can't figure
    # out how to make it work with numpy audio files
    # TODO make sure we can process huge bags of melody wavs and see what happens

    print(f"melodies: {melodies}")

    melody_waveform = None
    melody_sr = None
    melody = None

    processed_melodies = []
    target_sr = 32000
    target_ac = 1

    if melodies is not None:
        if len(melodies) == 1 and melodies[0] is not None:
            melody = melodies[0]
            melody_waveform, melody_sr = torchaudio.load(melody)

            melody_waveform = convert_audio(
                melody_waveform, melody_sr, target_sr, target_ac
            )

            melody_waveform = grab_audio_segment(melody_waveform, duration)

            # done in generate_with_chroma later?

            processed_melodies.append(melody_waveform)

    # print(f"processed_melodies at start: {processed_melodies}")
    # print(
    #    f"processed_melodies shape: {[m.shape if m is not None else None for m in processed_melodies]}"
    # )

    MODEL.set_generation_params(duration=duration, **gen_kwargs)
    print(
        "new batch",
        len(texts),
        texts,
    )

    if melodies is not None:
        for m in melodies:
            if m is not None:
                if isinstance(m, (str, bytes, os.PathLike)):
                    print(f"Melody string: {m}")
                else:
                    print(f"Melody numpy {m[0]},{m[1].shape}")

    be = time.time()

    full_continuation_wavform = None
    continuation_wavform = None
    continuation_wavform_sr = None

    continuation_wavform_duration = None
    if continuation_audio is not None and isinstance(
        continuation_audio, (str, bytes, os.PathLike)
    ):
        continuation_wavform, continuation_wavform_sr = torchaudio.load(
            continuation_audio
        )
        full_continuation_wavform = continuation_wavform.clone()

        extend_stride = gen_kwargs.get("extend_stride", 20)
        overlap = 30 - extend_stride
        print(f"overlap: {overlap} duration: {duration}")
        if overlap >= duration:
            overlap = duration - 1
        continuation_wavform = grab_audio_segment(continuation_wavform, overlap)

        continuation_wavform_duration = continuation_wavform.shape[-1] / 32000

    if len(melodies) > 1 and False:
        for melody in melodies:
            if melody is None:
                processed_melodies.append(None)
            else:
                sr, melody = (
                    melody[0],
                    torch.from_numpy(melody[1]).to(MODEL.device).float().t(),
                )
                if melody.dim() == 1:
                    melody = melody[None]
                melody = melody[..., : int(sr * duration)]
                melody = convert_audio(melody, sr, target_sr, target_ac)
                processed_melodies.append(melody)

    # print(f"processed_melodies at end: {processed_melodies}")
    if any(m is not None for m in processed_melodies):
        outputs = MODEL.generate_with_chroma_prompt(
            descriptions=texts,
            melody_wavs=processed_melodies,
            melody_sample_rate=target_sr,
            prompt=continuation_wavform,
            prompt_sample_rate=continuation_wavform_sr,
            progress=progress,
        )
    else:
        if continuation_wavform is not None:
            outputs = MODEL.generate_continuation(
                prompt=continuation_wavform,
                prompt_sample_rate=continuation_wavform_sr,
                descriptions=texts,
                progress=progress,
            )
        else:
            outputs = MODEL.generate(texts, progress=progress)

    outputs = outputs.detach().cpu().float()
    wav_files = []
    out_files = []
    date_str = datetime.datetime.now().strftime("%H%M_%S_%y%m%d")

    fileprefix = f"{model_name}_{date_str}"
    for output in outputs:
        # if full_continuation_wavform is not None:
        #    full_continuation_wavform = full_continuation_wavform.float()
        #    output = torch.cat(
        #        [full_continuation_wavform, output.detach().cpu().squeeze(0)], #dim=-1
        #    )
        with NamedTemporaryFile(
            "wb", prefix=fileprefix, suffix=".wav", delete=False
        ) as file:
            audio_write(
                file.name,
                output,
                MODEL.sample_rate,
                strategy="loudness",
                loudness_headroom_db=16,
                loudness_compressor=True,
                add_suffix=False,
            )

            style_changes = []

            continuation_seg_length = 30 - continuation_overlap

            current_wav_counter = 0

            """
            style_changes = [
                (2, {"bars_color": ("#fbbf24", "#ea580c")}),
                (4, {"bars_color": ("#FF0000", "#00FF00")}),
                (6, {"bars_color": ("#0000FF", "#FF00FF")}),
            ]
            """
            if continuation_wavform_duration is not None:
                # add style for 0 to length of continuation
                style_changes.append(
                    (
                        continuation_wavform_duration,
                        {
                            "bars_color": ("#fbbf24", "#ea580c"),
                        },
                    )
                )
                current_wav_counter += continuation_wavform_duration
            elif duration > 30:
                current_wav_counter += 30

            while current_wav_counter < duration:
                current_wav_counter += continuation_seg_length
                rng_c = random_color_bright()
                rng_c2 = random_color_bright()

                style_changes.append(
                    (
                        current_wav_counter,
                        {
                            "bars_color": (rng_c, rng_c2),
                        },
                    )
                )

            waveform_text = texts[0]
            out_files.append(
                pool.submit(
                    make_waveform_custom,
                    file.name,
                    fileprefix=fileprefix,
                    style_changes=style_changes,
                    text=waveform_text,
                )
            )
            wav_files.append(file.name)

    res = [out_file.result() for out_file in out_files]
    print("batch finished", len(texts), time.time() - be)
    return res, wav_files


from gradio import components, processing_utils, routes, utils
import numpy as np
import matplotlib.pyplot as plt

import PIL
import PIL.Image
from typing import TYPE_CHECKING, Any, Callable, Iterable


import random

from scipy.fft import fft
from scipy.signal import spectrogram
import matplotlib as mpl

from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from scipy.fft import fft
from scipy.signal import spectrogram
from matplotlib import cm

from scipy.special import expit  # sigmoid function


# messed up s
def audio_to_properties(
    filename,
    color_map_base=1000,
    bar_count_mean=75,
    bar_count_stddev=10,
    complexity_scaling_factor=5500,
):  # added complexity scaling factor
    """
    Parameters:
        filename: Audio file path
        color_map_base: Base for logarithmic mapping of color map (default is 1000)
        bar_count_mean: Mean for Gaussian distribution of bar count (default is 75)
        bar_count_stddev: Standard deviation for Gaussian distribution of bar count (default is 10)
        complexity_scaling_factor: Scaling factor for spectral complexity impact on bar count (default is 500)
    """

    # Load and process the audio file
    frame_rate, data = processing_utils.audio_from_file(filename)
    if len(data.shape) > 1:
        data = np.mean(data, 1)

    # Compute the Fourier transform
    fft_res = np.abs(fft(data))

    # Compute the spectrogram
    f, t, Sxx = spectrogram(data, frame_rate)

    # Compute the average and dominant frequencies
    avg_freq = np.sum(fft_res * np.arange(len(fft_res))) / np.sum(fft_res)
    dominant_freq = np.argmax(fft_res)

    # Apply logarithmic mapping to frequencies
    avg_freq = np.log10(avg_freq + 1) / np.log10(color_map_base)
    normalized_dom_freq = np.log10(dominant_freq + 1) / np.log10(color_map_base)

    # Compute the spectral complexity (number of local maxima in the spectrogram)
    spectral_complexity = np.sum(Sxx[1:-1] > np.maximum(Sxx[:-2], Sxx[2:]))

    # Compute bar count using Gaussian function, influenced by spectral complexity
    bar_count = (
        bar_count_mean
        + bar_count_stddev * np.random.randn()
        + spectral_complexity / complexity_scaling_factor
    )  # used complexity scaling factor

    # Ensure bar count stays within a reasonable range
    bar_count = np.clip(bar_count, 25, 100)  # capped at 100

    # Use sigmoid function to map normalized frequencies to colors
    bg_color = mpl.colormaps["rainbow"](expit(avg_freq))[:3]
    bars_color = mpl.colormaps["viridis"](expit(normalized_dom_freq))[:3]

    # Convert colors to hexadecimal format
    hex_bg_color = "#" + "".join([f"{int(x*255):02x}" for x in bg_color])
    hex_bars_color = "#" + "".join([f"{int(x*255):02x}" for x in bars_color])

    return hex_bg_color, hex_bars_color, int(bar_count)


def random_color_bright():
    bright_color_list = [
        "#F90D1B",
        "#FDE005",
        "#EC00FC",
        "#9D00FE",
        "#00CF35",
        "#77ECF2",
        "#2132FF",
        "#FF03E4",
        "#FFF40A",
        "#46C940",
        "#FFEB3B",
        "#1C3499",
        "#FDA33B",
        "#FFFF67",
        "#0046A8",
        "#D13F5C",
        "#F58A38",
        "#548538",
        "#313769",
        "#2FD1F5",
        "#29E305",
        "#FFE814",
        "#FF007F",
        "#FFE500",
        "#FF4198",
        "#02B784",
        "#06336E",
        "#926DBD",
        "#127CFC",
        "#000000",
        "#FFC814",
        "#FFDE14",
        "#DB1A1A",
        "#C92424",
        "#212121",
        "#009945",
        "#734C32",
        "#F87937",
        "#FE6C2B",
        "#6A0BA8",
        "#46C940",
        "#2EB5EA",
    ]

    return random.choice(bright_color_list)
    # rand = lambda: random.randint(200, 255)
    # return "#%02X%02X%02X" % (rand(), rand(), rand())


def make_waveform_custom(
    audio: str | tuple[int, np.ndarray],
    *,
    bg_color: str = "#f3f4f6",
    bg_image: str | None = None,
    fg_alpha: float = 0.75,
    bars_color: str | tuple[str, str] = ("#fbbf24", "#ea580c"),
    bar_count: int = 50,
    bar_width: float = 0.6,
    fileprefix: str = "musicgen_",
    style_changes: list[tuple[int, dict]] = None,
    text: str = None,
    even_more_text: int = 10,
    auto_calc_everything: bool = False,
    auto_calc_bars: bool = False,
):
    """
    Generates a waveform video from an audio file. Useful for creating an easy to share audio visualization. The output should be passed into a `gr.Video` component.
    Parameters:
        audio: Audio file path or tuple of (sample_rate, audio_data)
        bg_color: Background color of waveform (ignored if bg_image is provided)
        bg_image: Background image of waveform
        fg_alpha: Opacity of foreground waveform
        bars_color: Color of waveform bars. Can be a single color or a tuple of (start_color, end_color) of gradient
        bar_count: Number of bars in waveform
        bar_width: Width of bars in waveform. 1 represents full width, 0.5 represents half width, etc.
    Returns:
        A filepath to the output video.
    """

    og_text = text

    print(f"style_changes: {style_changes}")
    if isinstance(audio, str):
        audio_file = audio
        audio = processing_utils.audio_from_file(audio)
    else:
        tmp_wav = NamedTemporaryFile(suffix=".wav", delete=False)
        processing_utils.audio_to_file(audio[0], audio[1], tmp_wav.name, format="wav")
        audio_file = tmp_wav.name
    duration = round(len(audio[1]) / audio[0], 4)

    if auto_calc_everything is True:
        bg_color, bars_color, bar_count = audio_to_properties(audio_file)
    elif auto_calc_bars is True:
        bg_color_toss, bars_color_toss, bar_count = audio_to_properties(audio_file)

    def hex_to_rgb(textcolor):
        textcolor = textcolor.lstrip("#")
        try:
            rgb = tuple(int(textcolor[i : i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            rgb = (255, 255, 255)
        return rgb

    def get_color_gradient(c1, c2, n):
        assert n > 1
        c1_rgb = np.array(hex_to_rgb(c1)) / 255
        c2_rgb = np.array(hex_to_rgb(c2)) / 255
        mix_pcts = [x / (n - 1) for x in range(n)]
        rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
        return [
            "#" + "".join(f"{int(round(val * 255)):02x}" for val in item)
            for item in rgb_colors
        ]

    # Reshape audio to have a fixed number of bars
    samples = audio[1]
    if len(samples.shape) > 1:
        samples = np.mean(samples, 1)
    bins_to_pad = bar_count - (len(samples) % bar_count)
    samples = np.pad(samples, [(0, bins_to_pad)])
    samples = np.reshape(samples, (bar_count, -1))
    samples = np.abs(samples)

    samples = np.max(samples, 1)
    max_sample = np.max(samples)

    with utils.MatplotlibBackendMananger():
        plt.clf()
        # Plot waveform
        color = (
            bars_color
            if isinstance(bars_color, str)
            else get_color_gradient(bars_color[0], bars_color[1], bar_count)
        )

    # Compute the color change indices based on the style_changes parameter
    color_change_indices = None
    color_gradients = None
    if style_changes is not None:
        color_change_indices = [
            round(change[0] * bar_count / duration) for change in style_changes
        ]
        color_gradients = [
            get_color_gradient(
                change[1]["bars_color"][0],
                change[1]["bars_color"][1],
                max(2, bar_count - idx),
            )
            for idx, change in zip(color_change_indices, style_changes)
        ]

    j = -1  # Initialize j

    if text is not None and len(text) > 4:
        text = text * (bar_count)

    text_plt = None
    for i in range(bar_count):
        # Change the color if necessary
        gradient = None
        if color_change_indices is not None:
            for j, idx in enumerate(color_change_indices):
                if i >= idx:
                    bars_color = style_changes[j][1].get("bars_color", bars_color)
                    gradient = color_gradients[j]
                else:
                    break  # Exit the loop as soon as we find a segment that hasn't started yet

        # Determine the color of the current bar
        if isinstance(bars_color, str):
            color = bars_color
        else:
            color = (
                gradient[i - color_change_indices[j]]
                if gradient
                and j >= 0
                and j > len(color_change_indices)
                and (len(gradient) > i - color_change_indices[j])
                else get_color_gradient(bars_color[0], bars_color[1], bar_count)[i]
            )

        plt.bar(
            i,
            samples[i] * 2,
            bottom=(-1 * samples[i]),
            width=bar_width,
            color=color,
        )

        if text is not None and len(text) > 4 and i < len(samples):
            if i < len(text):
                # print(f"i: {i}, text[i]: {text[i]}")
                # fontsize = samples[i]

                text_height_ratio = samples[i] / max_sample
                # text_height_maybe = samples[i] / 32000
                fontsize = random.randint(5, 12) * text_height_ratio * 2.0
                fontweight = max(500 * text_height_ratio, 1000)
                # print(f"fontsize: {fontsize} text_height_maybe {text_height_ratio}")

                textcolor = bars_color
                # print(f"textcolor: {textcolor} is type {type(textcolor)}")
                if textcolor is not None and textcolor is not type(tuple):
                    # print(f"textcolor: {textcolor} is type {type(textcolor)}")

                    textcolor = cycle_color(textcolor)
                else:
                    textcolor = cycle_color(textcolor[0])

                rotation = random.randint(89, 92)

                text_to_show = text[i : i + even_more_text]
                # remove those from text variable
                text = text[i + even_more_text :]

                adjusted_y = 2 * samples[i] - (2 * max_sample)

                text_plt = plt.text(
                    x=i,
                    y=adjusted_y,  # change this line
                    s=text_to_show,
                    fontsize=fontsize,
                    color=textcolor,
                    fontweight=fontweight,
                    rotation=rotation,
                )

        plt.axis("off")
        plt.margins(x=0)
        tmp_img = NamedTemporaryFile(suffix=".png", delete=False)
        savefig_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if bg_image is not None:
            savefig_kwargs["transparent"] = True
        else:
            savefig_kwargs["facecolor"] = bg_color
        plt.savefig(tmp_img.name, **savefig_kwargs)
        waveform_img = PIL.Image.open(tmp_img.name)
        waveform_img = waveform_img.resize((1000, 200))

        # Composite waveform with background image
        if bg_image is not None:
            waveform_array = np.array(waveform_img)
            waveform_array[:, :, 3] = waveform_array[:, :, 3] * fg_alpha
            waveform_img = PIL.Image.fromarray(waveform_array)

            bg_img = PIL.Image.open(bg_image)
            waveform_width, waveform_height = waveform_img.size
            bg_width, bg_height = bg_img.size
            if waveform_width != bg_width:
                bg_img = bg_img.resize(
                    (waveform_width, 2 * int(bg_height * waveform_width / bg_width / 2))
                )
                bg_width, bg_height = bg_img.size
            composite_height = max(bg_height, waveform_height)
            composite = PIL.Image.new(
                "RGBA", (waveform_width, composite_height), "#FFFFFF"
            )
            composite.paste(bg_img, (0, composite_height - bg_height))
            composite.paste(
                waveform_img, (0, composite_height - waveform_height), waveform_img
            )
            composite.save(tmp_img.name)
            img_width, img_height = composite.size
        else:
            img_width, img_height = waveform_img.size
            waveform_img.save(tmp_img.name)

    # Convert waveform to video with ffmpeg
    output_mp4 = NamedTemporaryFile(suffix=".mp4", delete=False, prefix=fileprefix)

    sentence = og_text

    if (og_text is not None) and (len(og_text) > 4):
        min_words = 50
        x_offset_ratio = 0.02  # starting
        x_motion_ratio = (
            0.8  # pad to size of font, need to adjust this based on font size
        )
        y_amplitude_ratio = 1 / 4  # Sine wave amplitude
        y_amplitude_ratio = 1 / 3  # Sine wave amplitude
        # not really quite right sometimes ti's still off the screen

        y_const = 0.0
        y_offset_ratio = 1 / 4  # Sine wave offset
        outline_thickness = 0  # this blows up the ffmpeg filter length too much and the unix shell dies, which breaks my heart. IT LOOKS SO COOL. It gives you repeating popping YouTube style outlined words. Works okay with a few text only...

        words = sentence.split()

        while len(words) < min_words:
            words += words

        num_words = len(words)
        word_duration = duration / num_words
        filters = []
        # I don't know why I'm doing this... just amazed ffmpeg has all this stuff built in
        last_color = None
        color = random_color()
        for i, word in enumerate(words):
            start_time = i * word_duration
            end_time = (i + 1) * word_duration
            last_color = color
            color = random_color()

            for j in range(-outline_thickness - 1, outline_thickness + 1):
                for k in range(-outline_thickness - 1, outline_thickness + 1):
                    x_expr = f"'(t/{duration})*{img_width}*{x_motion_ratio}+{img_width}*{x_offset_ratio}+{j}'"
                    y_expr = f"'sin(2*PI*(t/{duration}+{i/num_words}))*{img_height}*{y_amplitude_ratio}+{img_height}*{y_offset_ratio}+{k}+{y_const}'"

                    filters.append(
                        f"drawtext=enable='between(t,{start_time},{end_time})':text='{word}':x={x_expr}:y={y_expr}:fontsize=34:fontcolor=#{last_color}"
                    )

            x_expr = f"'(t/{duration})*{img_width}*{x_motion_ratio}+{img_width}*{x_offset_ratio}'"
            y_expr = f"'sin(2*PI*(t/{duration}+{i/num_words}))*{img_height}*{y_amplitude_ratio}+{img_height}*{y_offset_ratio}+{y_const}'"
            filters.append(
                f"drawtext=enable='between(t,{start_time},{end_time})':text='{word}':x={x_expr}:y={y_expr}:fontsize=34:fontcolor=#{color}"
            )

        filters_str = ",".join(filters)

        ffmpeg_cmd = f"""ffmpeg -loop 1 -i {tmp_img.name} -i {audio_file} -vf "{filters_str}" -t {duration} -y {output_mp4.name}"""
    else:
        ffmpeg_cmd = f"""ffmpeg -loop 1 -i {tmp_img.name} -i {audio_file} -vf "color=c=#FFFFFF77:s={img_width}x{img_height}[bar];[0][bar]overlay=-w+(w/{duration})*t:H-h:shortest=1" -t {duration} -y {output_mp4.name}"""

    sp.call(ffmpeg_cmd, shell=True)
    return output_mp4.name


import colorsys


def hex_to_hsv(hexcolor):
    r, g, b = (int(hexcolor[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hsv(r, g, b)


def hsv_to_hex(h, s, v):
    r, g, b = (int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))
    return "#%02x%02x%02x" % (r, g, b)


def cycle_color(hexcolor):
    if isinstance(hexcolor, tuple):
        hexcolor = hexcolor[0]
    if len(hexcolor) != 7 or hexcolor[0] != "#":
        raise ValueError(
            f"Expected a color string of the format '#RRGGBB' but got {hexcolor}"
        )
    h, s, v = hex_to_hsv(hexcolor[1:])
    h += 0.1
    if h > 1:
        h -= 1
    return hsv_to_hex(h, s, v)


def invert_color(textcolor):
    if isinstance(textcolor, tuple):
        textcolor = textcolor[0]
    if len(textcolor) != 7 or textcolor[0] != "#":
        raise ValueError(
            f"Expected a color string of the format '#RRGGBB' but got {textcolor}"
        )

    textcolor = textcolor.lstrip("#")
    rgb = tuple(int(textcolor[i : i + 2], 16) for i in (0, 2, 4))
    inverted_color = "#%02x%02x%02x" % tuple([255 - x for x in rgb])
    return inverted_color


def random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f"{r:02x}{g:02x}{b:02x}"


def predict_batched(texts, melodies):
    max_text_length = 512
    texts = [text[:max_text_length] for text in texts]
    load_model("melody")
    res = _do_predictions(texts, melodies, BATCHED_DURATION)
    return [res, melodies]


def predict_full_log_first(
    model,
    text,
    melody,
    duration,
    topk,
    topp,
    temperature,
    cfg_coef,
    seed,
    continuation_overlap,
    continuation_audio,
    two_step_cfg,
    progress=gr.Progress(),
):
    parameters = locals().copy()

    del parameters["progress"]

    # more readable date, normal human style
    # datetime.datetime.now().strftime("%y%m%d_%H%M_%S")
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parameters["date"] = date_str

    parameters_json = json.dumps(parameters, default=str, indent=4)

    html = f"Current Status: {parameters_json}"
    return gr.update(value=html)


def predict_full(
    model,
    text,
    melody,
    duration,
    topk,
    topp,
    temperature,
    cfg_coef,
    seed,
    continuation_overlap,
    continuation_audio,
    two_step_cfg,
    progress=gr.Progress(),
):
    global INTERRUPTING
    INTERRUPTING = False
    # if temperature < 0:
    #    raise gr.Error("Temperature must be >= 0.")
    # if topk < 0:
    #    raise gr.Error("Topk must be non-negative.")
    # if topp < 0:
    #    raise gr.Error("Topp must be non-negative.")

    topk = int(topk)
    load_model(model)

    def _progress(generated, to_generate):
        progress((generated, to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")

    MODEL.set_custom_progress_callback(_progress)

    seed = parse_or_set_seed(seed, 0)

    parameters = locals().copy()

    del parameters["progress"]
    del parameters["_progress"]

    # more readable date, normal human style
    # datetime.datetime.now().strftime("%y%m%d_%H%M_%S")
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parameters["date"] = date_str

    extend_stride = 30 - continuation_overlap

    outs, wavs = _do_predictions(
        model,
        [text],
        [melody],
        duration,
        continuation_overlap,
        continuation_audio,
        seed,
        progress=True,
        top_k=topk,
        top_p=topp,
        temperature=temperature,
        cfg_coef=cfg_coef,
        extend_stride=extend_stride,
        two_step_cfg=two_step_cfg,
    )

    # return outs[0]

    file = outs[0]

    path_audio = pathlib.Path(file).resolve()

    path_settings = path_audio.parent / f"{path_audio.stem}.json"

    parameters["output_sample"] = wavs[0]

    delimiter = "##ENDSAMPLE##\n"
    parameters_json = json.dumps(parameters, default=str, indent=4) + delimiter

    parameters["output_sample"] = wavs[0]
    if not IN_GRADIO_SPACE:
        os.makedirs(output_dir, exist_ok=True)

    with path_settings.open("w", encoding="utf-8") as settings_file:
        settings_file.write(f"parameters_json: {parameters_json}")

    try:
        shutil.copy(path_audio, output_dir)
        shutil.copy(settings_file.name, output_dir)
        shutil.copy(wavs[0], output_dir)
        print(f"wrote samples to {output_dir}")
    except Exception as e:
        print("error copying files to output dir", e)

    return (
        outs[0],
        [wavs[0], str(path_settings), outs[0]],
        parameters_json,
    )


musicgen_css_style = """

body .musicgen_upload_audio, body .musicgen_upload_file {
  height: 140px !important;
}

.musicgen_upload_audio .svelte-19sk1im::before {
    content: "Click to Trim →" !important;
    position: absolute;
    left: -90px !important;  
    z-index: 9999;
  background-color: var(--secondary-800) !important;
  border-radius: 3px;
  padding: 1px;
}

body .musicgen_output_list {
  height: 160px !important;

}


.json_active {
  padding: 5px;
  border: 2px dashed orange;
  border-radius: 9px;
}


    .file-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px;
        border-bottom: 1px solid #ccc;
    }
    .summary-content {
        font-size: 0.8em;
        color: #888;
    }

.json_active.prose {
font-size: 0.9em
}


"""

css += musicgen_css_style


def truncate_text(text, length=55):
    return (text[:length] + "..") if len(text) > length else text


def update_output_files(
    output_video, output_files_history, json_out, output_files, debug=True
):
    table_html = ""
    if json_out is not None:
        # print(f"json_out: {json_out}")
        json_out = json_out.replace("parameters_json: ", "", 1).strip()

        delimiter = "##ENDSAMPLE##"
        json_out = json_out.strip().split(delimiter)[0]
        # actually just one element
        # print(f"json_out aa: {json_out}")

        json_dict = json.loads(json_out)
        output_video_filename_only = output_video.split("/")[-1]

        melody_filename_only = ""
        if (
            "melody" in json_dict
            and json_dict["melody"] is not None
            and len(json_dict["melody"]) > 0
        ):
            melody_filename_only = json_dict["melody"].split("/")[-1]

        output_files_history = output_files_history.replace("</tbody></table>", "")
        text_samp = truncate_text(json_dict["text"])
        print(f"text_samp: {text_samp}")
        table_row = (
            f"<tr class=\"tr-body svelte-13hsdno\" data-raw-json='{json_out}'>"
            f'<td class="svelte-13hsdno"><a href="file={output_video}" download>{output_video_filename_only}</a></td>'
            f'<td class="svelte-13hsdno"><div class="svelte-1ayixqk table">{json_dict["model"]} {json_dict["duration"]}s</div></td>'
            f'<td class="svelte-13hsdno"><div class="svelte-1ayixqk table">{text_samp}</div></td>'
            f'<td class="svelte-13hsdno"><div class="svelte-1ayixqk table">{melody_filename_only}</div></td>'
            f'<td class="svelte-13hsdno"><div class="svelte-1ayixqk table">{json_dict.get("continuation_audio", "")}</div></td>'
            f'<td class="svelte-13hsdno"><button class="details-button svelte-1ayixqk table">Details</button></td>'
            f"</tr></tbody></table>"
        )
        table_html += table_row

    total_history = output_files_history + table_html
    return gr.update(value=total_history)


operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}


def eval_expr(expr):
    """
    Safely evaluates a mathematical expression and returns the result.
    """
    return eval_(ast.parse(expr, mode="eval").body)


def eval_(node):
    if isinstance(node, ast.Num):  # <number>
        return node.n
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        return operators[type(node.op)](eval_(node.left), eval_(node.right))
    else:
        raise TypeError(node)


def update_from_json(json_input, json_dummy):
    # print(f"update_from_json: {json_input}")

    json_str = json_input.replace("parameters_json: ", "", 1).strip()
    parameters = json.loads(json_str)

    try:
        last_sample_parameters = json.loads(
            json_dummy.replace("parameters_json: ", "", 1).strip()
        )

        updated_components = []
        for key in parameters.keys():
            param_value = parameters[key]
            if isinstance(param_value, str) and param_value.startswith("LAST_SAMPLE_"):
                last_sample_key = param_value.split("LAST_SAMPLE_")[1]

                # Check for a mathematical operation
                if " / " in last_sample_key:
                    last_sample_key, operation = last_sample_key.split(" / ", 1)
                    # Safely evaluate the mathematical expression
                    value = eval_expr(
                        str(last_sample_parameters.get(last_sample_key, 0))
                        + " / "
                        + operation
                    )
                elif " * " in last_sample_key:
                    last_sample_key, operation = last_sample_key.split(" * ", 1)
                    value = eval_expr(
                        str(last_sample_parameters.get(last_sample_key, 0))
                        + " * "
                        + operation
                    )
                elif " - " in last_sample_key:
                    last_sample_key, operation = last_sample_key.split(" - ", 1)
                    value = eval_expr(
                        str(last_sample_parameters.get(last_sample_key, 0))
                        + " - "
                        + operation
                    )
                elif " + " in last_sample_key:
                    last_sample_key, operation = last_sample_key.split(" + ", 1)
                    value = eval_expr(
                        str(last_sample_parameters.get(last_sample_key, 0))
                        + " + "
                        + operation
                    )
                else:
                    # Substitute the last sample value
                    value = last_sample_parameters.get(last_sample_key)

                updated_components.append(gr.update(value=value))
            else:
                updated_components.append(gr.update(value=param_value))
        return updated_components
    except Exception as e:
        print(f"Error: {e}")
        return []


def start_music_sequence(json_input, json_dummy):
    # print(f"start_music_sequence: {json_input}")

    global GENERATION_SEQUENCES

    last_sample_parameters = {}

    try:
        last_sample_parameters = json.loads(
            json_dummy.replace("parameters_json: ", "", 1).strip()
        )
    except Exception as e:
        print(f"Error: {e}")

    if GENERATION_SEQUENCES is None:
        print(f"First time start")
        json_input = json_input.replace("parameters_json: ", "", 1).strip()

        delimiter = "##ENDSAMPLE##"
        json_objects = json_input.strip().split(delimiter)

        GENERATION_SEQUENCES = []
        for idx, obj in enumerate(json_objects):
            if obj.strip() != "":
                obj = obj.replace("parameters_json:", "", 1).strip()

                print(f"{idx} CREATING: {obj}")
                GENERATION_SEQUENCES.append((json.loads(obj), idx))
    else:
        print(f"Later start")

    if len(GENERATION_SEQUENCES) == 0:
        print(f"  >No more sequences")
        return []

    next_generation, sequence_number = GENERATION_SEQUENCES.pop(0)

    print(f"  Running Sequence: {sequence_number}: {next_generation}")
    try:
        updated_components = []
        for key in next_generation.keys():
            param_value = next_generation[key]
            if isinstance(param_value, str) and param_value.startswith("LAST_SAMPLE_"):
                last_sample_key = param_value.split("LAST_SAMPLE_")[1]

                if " / " in last_sample_key:
                    last_sample_key, operation = last_sample_key.split(" / ", 1)
                    value = eval_expr(
                        str(last_sample_parameters.get(last_sample_key, 0))
                        + " / "
                        + operation
                    )
                elif " * " in last_sample_key:
                    last_sample_key, operation = last_sample_key.split(" * ", 1)
                    value = eval_expr(
                        str(last_sample_parameters.get(last_sample_key, 0))
                        + " * "
                        + operation
                    )
                elif " - " in last_sample_key:
                    last_sample_key, operation = last_sample_key.split(" - ", 1)
                    value = eval_expr(
                        str(last_sample_parameters.get(last_sample_key, 0))
                        + " - "
                        + operation
                    )
                elif " + " in last_sample_key:
                    last_sample_key, operation = last_sample_key.split(" + ", 1)
                    value = eval_expr(
                        str(last_sample_parameters.get(last_sample_key, 0))
                        + " + "
                        + operation
                    )
                else:
                    # Substitute the last sample value
                    value = last_sample_parameters.get(last_sample_key)

                updated_components.append(gr.update(value=value))
            else:
                updated_components.append(gr.update(value=param_value))
        return updated_components
    except Exception as e:
        print(f"Error: {e}")
        return []


def ui_full(launch_kwargs):
    with gr.Blocks(css=css, title="MusicGen Quantum") as interface:
        with gr.Row():
            gr.Markdown(
                """
                # MusicGen - Spooky Rhythm at a Distance Edition
                This is your private demo for [MusicGen](https://github.com/facebookresearch/audiocraft), a simple and controllable model for music generation
                presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284)
                """
            )

        with gr.Row():
            with gr.Column(variant="primary"):
                with gr.Tab(
                    label="🎵 Generate Music",
                ) as musicgen_main_tab:
                    with gr.Row():
                        with gr.Column(variant="panel"):
                            gr.Markdown(
                                """
                            ## Three Types of Input: 📜 Text, 🎵 Melody, 🔊 Music
                            Each can describe the music. Use any combination, but 🎵 Melody requires melody model.
                            """
                            )
                            with gr.Row():
                                text = gr.Textbox(
                                    label="📜 Text: Describe the music in words.",
                                    interactive=True,
                                    info="",
                                    lines=3,
                                )

                                melody = gr.Audio(
                                    source="upload",
                                    type="filepath",
                                    label="🎵 Melody: Audio File as example melody.",
                                    interactive=True,
                                    elem_classes="musicgen_upload_audio",
                                    info="",
                                )
                            with gr.Row():
                                continuation_audio = gr.Audio(
                                    source="upload",
                                    type="filepath",
                                    label="🔊 Music: Audio File to continue from.",
                                    info="",
                                    interactive=True,
                                    elem_classes="musicgen_upload_audio",
                                )
                                generation_options = gr.CheckboxGroup(
                                    [
                                        "Match Position To Duration in Longer Melody File",
                                        # "Repeat Melody Wav Files If Shorter Than Total Duration",
                                        # "Generate With No Melody Guidance After Melody File",
                                    ],
                                    label="Melody Options (WIP)",
                                )

                            with gr.Row():
                                model = gr.Radio(
                                    ["melody", "medium", "small", "large"],
                                    label="Model",
                                    value="melody",
                                    interactive=True,
                                )

                    with gr.Row():
                        submit = gr.Button("Submit", variant="primary")
                        # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                        cancel_button = gr.Button("Interrupt", variant="stop")
                        cancel_button.click(
                            fn=interrupt,
                            queue=False,
                        )

                    with gr.Row():
                        with gr.Column(variant="panel"):
                            with gr.Row():
                                duration = gr.Slider(
                                    minimum=1,
                                    maximum=300,
                                    value=20,
                                    label="Generated Total Music Duration",
                                    interactive=True,
                                    info="",
                                )

                                continuation_overlap = gr.Slider(
                                    minimum=0,
                                    maximum=30,
                                    value=15,
                                    label="Overlap (Continuation and >30s Only)",
                                    interactive=True,
                                    info="",
                                )
                    with gr.Row(variant="panel"):
                        with gr.Column(scale=1):
                            with gr.Row():
                                cfg_coef = gr.Slider(
                                    label="Classifier Free Guidance",
                                    info="This is probably the knob you want to tweak the most. Lots of possibilities.",
                                    value=3.0,
                                    interactive=True,
                                    minimum=-5.0,
                                    maximum=10.0,
                                )

                                with gr.Column(scale=1):
                                    gr.Markdown(
                                        """
                                        ***Classifier Free Guidance*** is how strongly the audio tries to stock to your instructions or the melody. 3.0 is what they used for melody in conditioning in the training, but may try lower values for pure text I think. 

                                        ***Negative Guidance*** is best avoided. But it is interesting that that you do get something a bit like 'opposite' or your music description, sometimes. 
                                        """
                                    )

                            with gr.Row():
                                with gr.Accordion(
                                    label="Advanced Settings", open=False
                                ):
                                    with gr.Row():
                                        topk = gr.Slider(
                                            label="topk",
                                            value=250,
                                            minimum=-1000,
                                            maximum=1000,
                                            step=1,
                                        )
                                        topp = gr.Slider(
                                            label="topp",
                                            value=0.0,
                                            minimum=-5.0,
                                            maximum=2.0,
                                        )
                                    with gr.Row():
                                        temperature = gr.Slider(
                                            label="temperature: ",
                                            info="",
                                            minimum=0.000,
                                            maximum=3.0,
                                            value=1.00,
                                            interactive=True,
                                        )
                                        seed = gr.Number(
                                            label="Seed", value=-1, interactive=True
                                        )

                                        match_melody_at = gr.Number(
                                            label="Match Melody at This Time Step",
                                            value=-1,
                                            interactive=True,
                                            info="WIP",
                                        )

                                        two_step_cfg = gr.Checkbox(
                                            label="Two Step Classifier Guidance",
                                            info="Slower, probably not any better",
                                            value=False,
                                            visible=False,
                                            interactive=False,
                                        )
                with gr.Tab(label=" 🎼📝✍ Music Sequences") as musicgen_advanced_tab:
                    with gr.Row():
                        gr.Markdown(
                            """
                            This is just a queue, but if you are granular enough with your commands, it's still pretty useful. 
                            """
                        )
                    with gr.Row():
                        json_input_data = gr.Text(
                            label="JSON",
                            lines=3,
                            interactive=True,
                            elem_id="text-input",
                            visible=True,
                        )
                    with gr.Row():
                        Generation_Sequence_Button = gr.Button(
                            "Run Sequence", variant="primary"
                        )

                with gr.Tab(label="  🌅🐳🎷🔫 Flux Capacitors") as musicgen_advanced_tab:
                    with gr.Row():
                        gr.Markdown(
                            """
                            WIP 
                            """
                        )

            with gr.Column():
                with gr.Row():
                    output_video = gr.Video(label="Generated Music")
                with gr.Row():
                    output_files = gr.File(
                        label="Save Generated Files",
                        interactive=False,
                        elem_classes="musicgen_output_list",
                    )

                with gr.Row():
                    json_before_gen = gr.HTML(
                        label="Status",
                        interactive=False,
                        elem_classes="json_active",
                        value="",
                        visible=True,
                    )

                with gr.Row(variant="panel"):
                    output_files_history = gr.HTML(
                        label="History",
                        interactive=False,
                        elem_classes="musicgen_output_history",
                        value=f"""<table class="svelte-13hsdno">
    <thead>
        <tr class="tr-head svelte-13hsdno">
            <th class="svelte-13hsdno">File</th>
            <th class="svelte-13hsdno">Model</th>
            <th class="svelte-13hsdno">Text</th>

            <th class="svelte-13hsdno">Melody</th>
            <th class="svelte-13hsdno">Continue</th>
        </tr>
    </thead>
    <tbody>""",
                        info="",
                    )

                with gr.Row():
                    json_dummy = gr.Text(
                        label="History",
                        interactive=False,
                        elem_classes="json_dummy",
                        value="",
                        visible=False,
                    )
        """
        sequence_start_event = output_files_history.change(
            start_music_sequence,
            inputs=[
                json_input_data,
                json_dummy,
            ],
            outputs=[
                model,
                text,
                melody,
                duration,
                topk,
                topp,
                temperature,
                cfg_coef,
                seed,
                continuation_overlap,
                continuation_audio,
                two_step_cfg,
            ],
        )
       
        sequence_start_event_success_event = sequence_start_event.success(
            predict_full_log_first,
            inputs=[
                model,
                text,
                melody,
                duration,
                topk,
                topp,
                temperature,
                cfg_coef,
                seed,
                continuation_overlap,
                continuation_audio,
                two_step_cfg,
            ],
            outputs=[json_before_gen],
            queue=None,
        )

        sequence_predict_full_event = sequence_start_event_success_event.success(
            predict_full,
            inputs=[
                model,
                text,
                melody,
                duration,
                topk,
                topp,
                temperature,
                cfg_coef,
                seed,
                continuation_overlap,
                continuation_audio,
                two_step_cfg,
            ],
            outputs=[output_video, output_files, json_dummy],
        )

        sequence_predict_output_event = sequence_predict_full_event.success(
            update_output_files,
            inputs=[output_video, output_files_history, json_dummy, output_files],
            outputs=[output_files_history],
        ).then(
            start_music_sequence,
            inputs=[
                json_input_data,
                json_dummy,
            ],
            outputs=[
                model,
                text,
                melody,
                duration,
                topk,
                topp,
                temperature,
                cfg_coef,
                seed,
                continuation_overlap,
                continuation_audio,
                two_step_cfg,
            ],
        )
        
        first_submit_event = submit.click(
            predict_full,
            inputs=[
                model,
                text,
                melody,
                duration,
                topk,
                topp,
                temperature,
                cfg_coef,
                seed,
                continuation_overlap,
                continuation_audio,
                two_step_cfg,
            ],
            outputs=[output_video, output_files, json_dummy],
        ).success(
            update_output_files,
            inputs=[output_video, output_files_history, json_dummy, output_files],
            outputs=[output_files_history],
        )
        """
        first_submit_event = submit.click(
            predict_full_log_first,
            inputs=[
                model,
                text,
                melody,
                duration,
                topk,
                topp,
                temperature,
                cfg_coef,
                seed,
                continuation_overlap,
                continuation_audio,
                two_step_cfg,
            ],
            outputs=[json_before_gen],
            queue=None,
        )

        first_submit_event_then = first_submit_event.then(
            predict_full,
            inputs=[
                model,
                text,
                melody,
                duration,
                topk,
                topp,
                temperature,
                cfg_coef,
                seed,
                continuation_overlap,
                continuation_audio,
                two_step_cfg,
            ],
            outputs=[output_video, output_files, json_dummy],
        ).success(
            update_output_files,
            inputs=[output_video, output_files_history, json_dummy, output_files],
            outputs=[output_files_history],
        )

        gr.Examples(
            fn=predict_full,
            examples=[
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
                [
                    "A jovial pirate shanty with hearty group vocals, buoyant banjo plucking, and rhythmic handclapping, providing an irresistible invitation to join the rollicking camaraderie.",
                    None,
                    "large",
                ],
            ],
            inputs=[text, melody, model],
            outputs=[output_video, output_files],
        )
        gr.Markdown(
            """
            ### More details

            The model will generate a short music extract based on the description you provided.
            The model can generate up to 30 seconds of audio in one pass. It is now possible
            to extend the generation by feeding back the end of the previous chunk of audio.
            This can take a long time, and the model might lose consistency. The model might also
            decide at arbitrary positions that the song ends.

            **WARNING:** Choosing long durations will take a long time to generate (2min might take ~10min). An overlap of 12 seconds
            is kept with the previously generated chunk, and 18 "new" seconds are generated each time.

            We present 4 model variations:
            1. Melody -- a music generation model capable of generating music condition on text and melody inputs. **Note**, you can also use text only.
            2. Small -- a 300M transformer decoder conditioned on text only.
            3. Medium -- a 1.5B transformer decoder conditioned on text only.
            4. Large -- a 3.3B transformer decoder conditioned on text only (might OOM for the longest sequences.)

            When using `melody`, ou can optionaly provide a reference audio from
            which a broad melody will be extracted. The model will then try to follow both the description and melody provided.

            You can also use your own GPU or a Google Colab by following the instructions on our repo.
            See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
            for more details.
            """
        )

        interface.queue().launch(**launch_kwargs)


def ui_batched(launch_kwargs):
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # MusicGen

            This is the demo for [MusicGen](https://github.com/facebookresearch/audiocraft), a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284).
            <br/>
            <a href="https://huggingface.co/spaces/facebook/MusicGen?duplicate=true" style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
            <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
            for longer sequences, more control and no queue.</p>
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(
                        label="Describe your music",
                        lines=3,
                        interactive=True,
                        elem_id="text-input",
                    )

                    melody = gr.Audio(
                        source="upload",
                        type="numpy",
                        label="Condition on a melody (optional)",
                        interactive=True,
                    )
                with gr.Row():
                    submit = gr.Button("Generate")
            with gr.Column():
                output = gr.Video(label="Generated Music", elem_id="generated-video")
                output_melody = gr.Audio(
                    label="Melody ", elem_id="melody-output", visible=False
                )

                with gr.Row():
                    output_files = gr.File(
                        label="Save Generated Files", interactive=False
                    )
                with gr.Row(visible=False) as share_row:
                    with gr.Group(elem_id="share-btn-container"):
                        gr.HTML(community_icon_html)
                        gr.HTML(loading_icon_html)
                        share_button = gr.Button(
                            "Share to community", elem_id="share-btn"
                        )
                        share_button.click(None, [], [], _js=share_js)
        submit.click(
            fn=lambda x: gr.update(visible=False),
            inputs=None,
            outputs=[share_row],
            queue=False,
            show_progress=False,
        ).then(
            predict_batched,
            inputs=[text, melody],
            outputs=[output, output_melody],
            batch=True,
            max_batch_size=MAX_BATCH_SIZE,
        ).then(
            fn=lambda x: gr.update(visible=True),
            inputs=None,
            outputs=[share_row],
            queue=False,
            show_progress=False,
        )
        gr.Examples(
            fn=predict_batched,
            examples=[
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                ],
                [
                    "A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                ],
                [
                    "90s rock song with electric guitar and heavy drums",
                    None,
                ],
                [
                    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130",
                    "./assets/bach.mp3",
                ],
                [
                    "lofi slow bpm electro chill with organic samples",
                    None,
                ],
                [
                    "A jovial pirate shanty with hearty group vocals, buoyant banjo plucking, and rhythmic handclapping, providing an irresistible invitation to join the rollicking camaraderie.",
                    None,
                    "large",
                ],
            ],
            inputs=[text, melody],
            outputs=[output_video, output_files],
        )
        gr.Markdown(
            """
        ### More details

        The model will generate 12 seconds of audio based on the description you provided.
        You can optionaly provide a reference audio from which a broad melody will be extracted.
        The model will then try to follow both the description and melody provided.
        All samples are generated with the `melody` model.

        You can also use your own GPU or a Google Colab by following the instructions on our repo.

        See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
        for more details.
        """
        )

        demo.queue(max_size=8 * 4).launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--listen",
        type=str,
        default="0.0.0.0" if "SPACE_ID" in os.environ else "127.0.0.1",
        help="IP to listen on for connections to Gradio",
    )
    parser.add_argument(
        "--username", type=str, default="", help="Username for authentication"
    )
    parser.add_argument(
        "--password", type=str, default="", help="Password for authentication"
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=0,
        help="Port to run the server listener on",
    )
    parser.add_argument("--inbrowser", action="store_true", help="Open in browser")
    parser.add_argument("--share", action="store_true", help="Share the gradio UI")
    parser.add_argument(
        "--not_in_space",
        action="store_true",
        help="If you aren't running this in a gradio space, you can set this flag and write files to a local directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="musicgen_samples",
        help="If you aren't running this in a gradio space, set output",
    )

    args = parser.parse_args()

    if args.not_in_space:
        IN_GRADIO_SPACE = False

        if args.output_dir:
            where_am_i = os.getcwd()
            output_dir = os.path.join(where_am_i, f"{args.output_dir}")
        else:
            output_dir = os.path.join(where_am_i, "musicgen_samples")

    launch_kwargs = {}
    launch_kwargs["server_name"] = args.listen

    if args.username and args.password:
        launch_kwargs["auth"] = (args.username, args.password)
    if args.server_port:
        launch_kwargs["server_port"] = args.server_port
    if args.inbrowser:
        launch_kwargs["inbrowser"] = args.inbrowser
    if args.share:
        launch_kwargs["share"] = args.share

    # Show the interface
    if IS_BATCHED:
        ui_batched(launch_kwargs)
    else:
        ui_full(launch_kwargs)
