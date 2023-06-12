import torch

import datetime

import platform
import sys

import re


def gpu_status_report(quick=False, gpu_no_details=False):
    status_report_string = ""

    if torch.cuda.is_available():
        device = torch.device("cuda")

        if gpu_no_details:
            status_report_string += f"{torch.cuda.get_device_name(device)}\n"
        else:
            status_report_string += "=== GPU Information ===\n"
            status_report_string += (
                f"GPU Device: {torch.cuda.get_device_name(device)}\n"
            )
            if not quick:
                status_report_string += f"Number of GPUs: {torch.cuda.device_count()}\n"
                status_report_string += (
                    f"Current GPU id: {torch.cuda.current_device()}\n"
                )
                status_report_string += (
                    f"GPU Capability: {torch.cuda.get_device_capability(device)}\n"
                )
                status_report_string += f"Supports Tensor Cores: {torch.cuda.get_device_properties(device).major >= 7}\n"

            props = torch.cuda.get_device_properties(device)
            status_report_string += (
                f"Total memory: {props.total_memory / (1024 ** 3)} GB\n"
            )

            status_report_string += f"CUDA Version: {torch.version.cuda}\n"
            status_report_string += f"PyTorch Version: {torch.__version__}\n"

    else:
        if gpu_no_details:
            status_report_string += "CPU or non CUDA device.\n"
        else:
            status_report_string += "No CUDA device is detected.\n"

    return status_report_string


def sanitize_filename(filename):
    return re.sub(r"[^a-zA-Z0-9_]", "_", filename)


def generate_unique_filepath(output_directory, filename, extension):
    filename = filename[:20]

    if not isinstance(filename, str) or not filename:
        raise ValueError("Filename should be a non-empty string.")

    if not isinstance(output_directory, str) or not output_directory:
        raise ValueError("Output directory should be a non-empty string.")

    base = sanitize_filename(filename)
    date_str = datetime.datetime.now().strftime("%y%m%d_%H%M_%S")
    unique_filename = f"{base}_{date_str}"

    # create directory if not exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    i = 0
    while os.path.isfile(os.path.join(output_directory, unique_filename + extension)):
        unique_filename = f"{base}_{date_str}_{i}"
        i += 1

    return os.path.join(output_directory, unique_filename)


def write_audio_file(output_directory, output, sample_rate, text):
    if not text:
        text = "no_text_prompt"

    try:
        output_filename = generate_unique_filepath(output_directory, text, ".wav")
        wavfile.write(
            output_filename + ".wav", sample_rate, np.array(output, dtype=np.float32)
        )
        print(f"File saved to {output_filename + '.wav'}")
    except Exception as e:
        print(f"Error occurred while writing file: {str(e)}")
        return None

    return output_filename  # Return without extension
