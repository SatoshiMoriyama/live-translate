"""CLIエントリポイント.

AudioCapture → transcribe_audio → translate → print のパイプラインを制御する。
VADにより発話区間を自動検出し、自然な文の区切りで処理する。
"""

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


def main() -> None:
    """メインループ. 音声キャプチャ→文字起こし→翻訳→表示を繰り返す."""
    translate_client = create_translate_client()

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
                print(f"[VAD] 発話検出 ({duration:.1f}秒)", flush=True)

                text = transcribe_audio(
                    audio=chunk,
                    language=LANGUAGE,
                    mlx_whisper_module=mlx_whisper,
                )
                if text and not is_hallucination(text):
                    ja_text = translate_text(
                        text=text,
                        source_lang="en",
                        target_lang="ja",
                        client=translate_client,
                    )
                    print(f"[EN] {text}", flush=True)
                    print(f"[JA] {ja_text}", flush=True)
                    print("---", flush=True)
        except KeyboardInterrupt:
            pass
