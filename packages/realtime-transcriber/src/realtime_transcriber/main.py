"""CLIエントリポイント.

AudioCapture → transcribe_audio → print のパイプラインを制御する。
"""

import time

import numpy as np
import sounddevice as sd

import mlx_whisper

from realtime_transcriber.audio import AudioCapture, is_silent
from realtime_transcriber.transcriber import is_hallucination, transcribe_audio

DEVICE_NAME = "BlackHole 2ch"
SAMPLE_RATE = 16000
BUFFER_SECONDS = 3
LANGUAGE = "en"
SLEEP_SECONDS = 0.1


def main() -> None:
    """メインループ. 音声キャプチャ→文字起こし→表示を繰り返す."""
    with AudioCapture(
        device_name=DEVICE_NAME,
        sample_rate=SAMPLE_RATE,
        buffer_seconds=BUFFER_SECONDS,
        sd_module=sd,
    ) as capture:
        try:
            while True:
                chunk = capture.get_audio_chunk()
                if chunk is None:
                    time.sleep(SLEEP_SECONDS)
                    continue

                if is_silent(chunk):
                    print(
                        f"[DEBUG] silent skip (rms={float(np.sqrt(np.mean(chunk**2))):.6f})",
                        flush=True,
                    )
                    continue

                text = transcribe_audio(
                    audio=chunk,
                    language=LANGUAGE,
                    mlx_whisper_module=mlx_whisper,
                )
                if text and not is_hallucination(text):
                    print(text, flush=True)
        except KeyboardInterrupt:
            pass
