"""CLIエントリポイント.

AudioCapture → transcribe_audio → translate → print のパイプラインを制御する。
VADにより発話区間を自動検出し、自然な文の区切りで処理する。
"""

import logging
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sounddevice as sd

import mlx_whisper

from realtime_transcriber.audio import AudioCapture
from realtime_transcriber.transcriber import is_hallucination, transcribe_audio
from realtime_transcriber.translator import create_translate_client, translate_text

DEVICE_NAME = "BlackHole 2ch"
SAMPLE_RATE = 16000
LANGUAGE = "en"
SLEEP_SECONDS = 0.05
MAX_PENDING_SECONDS = 15

logger = logging.getLogger(__name__)


def _is_sentence_end(text: str) -> bool:
    """テキストが文末（. ! ? など）で終わっているか判定する."""
    stripped = text.rstrip()
    if not stripped:
        return False
    # "..." は省略記号なので文末とみなさない
    if stripped.endswith("..."):
        return False
    return stripped[-1] in ".!?;"


_ABBREVIATIONS = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|etc|vs|e\.g|i\.e|al|Gen|Gov|Sgt|Corp)\.",
    re.IGNORECASE,
)
_ABBR_PLACEHOLDER = "\x00"


def _split_sentences(text: str) -> list[str]:
    """テキストを文単位に分割する."""
    # 略語のピリオドを一時的に退避して誤分割を防ぐ
    protected = _ABBREVIATIONS.sub(
        lambda m: m.group()[:-1] + _ABBR_PLACEHOLDER, text.strip()
    )
    sentences = re.split(r'(?<=[.!?])\s+', protected)
    return [s.replace(_ABBR_PLACEHOLDER, ".") for s in sentences if s.strip()]


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
    pending_audio: np.ndarray | None = None

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

                # 前回の未完結音声があれば結合
                if pending_audio is not None:
                    chunk = np.concatenate([pending_audio, chunk])
                    pending_audio = None

                duration = len(chunk) / SAMPLE_RATE

                text = transcribe_audio(
                    audio=chunk,
                    language=LANGUAGE,
                    mlx_whisper_module=mlx_whisper,
                    initial_prompt=prev_text or None,
                )
                if not text or is_hallucination(text):
                    continue

                # 文が完結していない & まだ蓄積に余裕がある場合は次のチャンクと結合
                if not _is_sentence_end(text) and duration < MAX_PENDING_SECONDS:
                    pending_audio = chunk
                    print(f"\r\033[90m  ... waiting\033[0m", end="", flush=True)
                    continue

                print("\r\033[K", end="", flush=True)  # pending表示をクリア
                # 直近の完全な文をコンテキストとして保持（文の途中で切れないように）
                sentences = _split_sentences(text)
                # 末尾から200文字以内に収まる文を取得
                context_parts: list[str] = []
                char_count = 0
                for s in reversed(sentences):
                    if char_count + len(s) > 200:
                        break
                    context_parts.append(s)
                    char_count += len(s)
                prev_text = " ".join(reversed(context_parts))

                def _translate_one(s: str) -> str:
                    try:
                        return translate_text(
                            text=s,
                            source_lang="en",
                            target_lang="ja",
                            client=translate_client,
                        )
                    except Exception:
                        logger.exception("Translation failed")
                        return "(翻訳失敗)"

                with ThreadPoolExecutor(max_workers=len(sentences)) as pool:
                    translated = list(pool.map(_translate_one, sentences))
                for sentence, ja_text in zip(sentences, translated):
                    print(f"\033[90m  {sentence}\033[0m", flush=True)
                    print(f"  {ja_text}", flush=True)
                print("", flush=True)
        except KeyboardInterrupt:
            pass
