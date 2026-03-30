"""CLIエントリポイント.

AudioCapture → transcribe_audio → translate → print のパイプラインを制御する。
VADにより発話区間を自動検出し、自然な文の区切りで処理する。
"""

import subprocess
import time

import sounddevice as sd

import mlx_whisper

from realtime_transcriber.audio import AudioCapture
from realtime_transcriber.transcriber import is_hallucination, transcribe_audio
from realtime_transcriber.translator import create_translate_client, translate_text

DEVICE_NAME = "BlackHole 2ch"
SAMPLE_RATE = 16000
LANGUAGE = "en"
SLEEP_SECONDS = 0.05


def _check_audio_output() -> None:
    """起動時に音声出力先の確認を促す."""
    default_out = sd.default.device[1]
    device_name = sd.query_devices(default_out)["name"]
    print(f"Output device: {device_name}")

    if "複数出力" not in device_name and "multi" not in device_name.lower():
        print("⚠ Please switch output to Multi-Output Device.")
        print("  Opening Sound Settings...")
        subprocess.run(
            ["open", "x-apple.systempreferences:com.apple.Sound-Settings.extension"],
        )
        input("  Press Enter after switching: ")


def main() -> None:
    """メインループ. 音声キャプチャ→文字起こし→翻訳→表示を繰り返す."""
    _check_audio_output()
    translate_client = create_translate_client()
    prev_text = ""

    with AudioCapture(
        device_name=DEVICE_NAME,
        sample_rate=SAMPLE_RATE,
        sd_module=sd,
    ) as capture:
        try:
            while True:
                chunk = capture.get_audio_chunk()
                if chunk is None:
                    time.sleep(SLEEP_SECONDS)
                    continue

                duration = len(chunk) / SAMPLE_RATE
                print(f"[VAD] Speech detected ({duration:.1f}s)", flush=True)

                text = transcribe_audio(
                    audio=chunk,
                    language=LANGUAGE,
                    mlx_whisper_module=mlx_whisper,
                    initial_prompt=prev_text or None,
                )
                if text and not is_hallucination(text):
                    prev_text = text[-200:]
                    ja_text = translate_text(
                        text=text,
                        source_lang="en",
                        target_lang="ja",
                        client=translate_client,
                    )
                    print(f"\033[90m  {text}\033[0m", flush=True)
                    print(f"\033[1;97m  {ja_text}\033[0m", flush=True)
                    print("", flush=True)
        except KeyboardInterrupt:
            pass
