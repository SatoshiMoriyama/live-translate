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
from realtime_transcriber.session_logger import SessionLogger
from realtime_transcriber.summarizer import Summarizer
from realtime_transcriber.transcriber import is_hallucination, transcribe_audio
from realtime_transcriber.translator import create_translate_client, translate_text

# --- 設定 ---
DEVICE_NAME = "BlackHole 2ch"
SAMPLE_RATE = 16000
LANGUAGE = "en"
SLEEP_SECONDS = 0.05
# 未完結の文を蓄積する最大秒数（これを超えたら未完結でも翻訳に回す）
MAX_PENDING_SECONDS = 15
# prev_text（Whisperコンテキスト）に保持する最大文字数
MAX_CONTEXT_CHARS = 200

# --- ANSIエスケープ ---
_DIM = "\033[90m"  # グレー文字（原文表示用）
_RESET = "\033[0m"
_CLEAR_LINE = "\r\033[K"  # カーソル行をクリア

logger = logging.getLogger(__name__)

# --- 文分割ユーティリティ ---

_ABBREVIATIONS = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|etc|vs|e\.g|i\.e|al|Gen|Gov|Sgt|Corp)\.",
    re.IGNORECASE,
)
_ABBR_PLACEHOLDER = "\x00"


def _is_sentence_end(text: str) -> bool:
    """テキストが文末（. ! ? など）で終わっているか判定する.

    省略記号 "..." は文末とみなさない。
    """
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped.endswith("..."):
        return False
    return stripped[-1] in ".!?;"


def _split_sentences(text: str) -> list[str]:
    """テキストを文単位に分割する.

    略語（Mr. Dr. e.g. など）のピリオドでは分割しない。
    """
    # 略語のピリオドをプレースホルダに退避して誤分割を防ぐ
    protected = _ABBREVIATIONS.sub(
        lambda m: m.group()[:-1] + _ABBR_PLACEHOLDER, text.strip()
    )
    sentences = re.split(r'(?<=[.!?])\s+', protected)
    # プレースホルダをピリオドに復元して返す
    return [s.replace(_ABBR_PLACEHOLDER, ".") for s in sentences if s.strip()]


# --- メインループのサブ処理 ---


def _check_audio_output() -> None:
    """起動時に音声出力先を確認し、Multi-Output Deviceでなければ切替を促す."""
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


def _build_context(sentences: list[str]) -> str:
    """直近の文から Whisper の initial_prompt 用コンテキストを組み立てる.

    末尾の文から逆順にたどり、MAX_CONTEXT_CHARS 以内に収まる範囲を返す。
    文の途中で切れないようにするため、文単位で取得する。
    """
    parts: list[str] = []
    char_count = 0
    for s in reversed(sentences):
        if char_count + len(s) > MAX_CONTEXT_CHARS:
            break
        parts.append(s)
        char_count += len(s)
    return " ".join(reversed(parts))


def _translate_sentences(
    sentences: list[str],
    translate_client: object,
) -> list[str]:
    """複数の文を並列で翻訳し、入力と同じ順序で翻訳結果を返す."""

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
        return list(pool.map(_translate_one, sentences))


def _print_results(
    sentences: list[str],
    translated: list[str],
    session_logger: SessionLogger,
) -> None:
    """原文（グレー）と翻訳文を1文ずつ表示し、ログに記録する."""
    for sentence, ja_text in zip(sentences, translated):
        print(f"{_DIM}  {sentence}{_RESET}", flush=True)
        print(f"  {ja_text}", flush=True)
        session_logger.log(sentence, ja_text)
    print("", flush=True)


# --- エントリポイント ---


def main() -> None:
    """メインループ. 音声キャプチャ→文字起こし→翻訳→表示を繰り返す."""
    _check_audio_output()
    translate_client = create_translate_client()
    session_logger = SessionLogger()
    summarizer = Summarizer(session_logger)
    summarizer.start()
    print(f"Log: {session_logger.path}")
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

                # 前回の未完結音声があれば先頭に結合
                if pending_audio is not None:
                    chunk = np.concatenate([pending_audio, chunk])
                    pending_audio = None

                duration = len(chunk) / SAMPLE_RATE

                # 要約をドメインヒントとして活用し、認識精度を向上させる
                summary_hint = summarizer.latest_summary
                if summary_hint and prev_text:
                    initial_prompt = summary_hint + " " + prev_text
                else:
                    initial_prompt = summary_hint or prev_text or None

                text = transcribe_audio(
                    audio=chunk,
                    language=LANGUAGE,
                    mlx_whisper_module=mlx_whisper,
                    initial_prompt=initial_prompt,
                )
                if not text or is_hallucination(text):
                    continue

                # 文が完結していない & 蓄積に余裕がある → 次のチャンクと結合して再処理
                if not _is_sentence_end(text) and duration < MAX_PENDING_SECONDS:
                    pending_audio = chunk
                    print(f"\r{_DIM}  ... waiting{_RESET}", end="", flush=True)
                    continue

                # pending 表示をクリアして結果を出力
                print(_CLEAR_LINE, end="", flush=True)
                sentences = _split_sentences(text)
                prev_text = _build_context(sentences)
                translated = _translate_sentences(sentences, translate_client)
                _print_results(sentences, translated, session_logger)
        except KeyboardInterrupt:
            summarizer.stop()
