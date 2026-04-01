"""文字起こしモジュール.

mlx-whisperによる音声テキスト変換を担当する。
"""

import re
from types import ModuleType

import numpy as np

# Apple Silicon向けに最適化されたWhisperモデル
# - "mlx-community/whisper-large-v3-turbo"（float16、約1.5GB、高精度）
# - "mlx-community/whisper-large-v3-turbo-q4"（4bit量子化、約400MB、高速）
MODEL_REPO = "mlx-community/whisper-large-v3-turbo-q4"

# Whisperが無音や短い音声に対して出力しがちな定型フレーズ
HALLUCINATION_PATTERNS = frozenset(
    {
        "thank you.",
        "thanks for watching.",
        "bye.",
        "bye!",
        "the end.",
        "subtitles by the amara.org community",
        "you",
        "thank you for watching.",
        "thank you so much for watching.",
        "thanks for watching!",
        "thank you for watching!",
    }
)


def _has_repetition(text: str, min_repeats: int = 5) -> bool:
    """テキストに同じ単語/フレーズの異常な繰り返しがあるか検出する."""
    # 同じ単語が連続で繰り返されるパターン（例: "too too too too too"）
    if re.search(r"\b(\w+)(?:\s+\1){" + str(min_repeats - 1) + r",}", text.lower()):
        return True
    # 同じ文字が連続で繰り返されるパターン（例: "llllllllll"）
    if re.search(r"(.)\1{9,}", text):
        return True
    # 同じ2〜6文字のパターンが5回以上繰り返される（例: "結論の結論の結論の..."）
    if re.search(r"(.{2,6})\1{4,}", text):
        return True
    return False


def _clean_repetition(text: str) -> str:
    """テキストから異常な繰り返し部分を除去する."""
    # 同じ2〜6文字のパターンの5回以上の連続を1回に置換
    cleaned = re.sub(r"(.{2,6})\1{4,}", r"\1", text)
    # 同じ文字の10回以上の連続を1文字に置換（例: "lllllll" → "l"）
    cleaned = re.sub(r"(.)\1{9,}", r"\1", cleaned)
    # 同じ単語の5回以上の連続を1回に置換
    cleaned = re.sub(r"\b(\w+)(?:\s+\1){4,}", r"\1", cleaned)
    return cleaned.strip()


def is_hallucination(text: str) -> bool:
    """Whisperの出力テキストが既知のハルシネーションパターンに一致するか判定する."""
    normalized = text.strip().lower()
    if not normalized:
        return False
    # 英数字がほぼ含まれないテキスト（例: "....", "...")はノイズ
    if not re.search(r"[a-zA-Z]{2,}", normalized):
        return True
    if normalized in HALLUCINATION_PATTERNS:
        return True
    # 繰り返しが大半を占める場合はハルシネーション
    # （例: "too too too too too too" → 除去後が元の50%未満ならハルシネーション）
    if _has_repetition(normalized):
        cleaned = _clean_repetition(normalized)
        if len(cleaned) < len(normalized) * 0.5:
            return True
    return False


def transcribe_audio(
    audio: np.ndarray,
    language: str,
    mlx_whisper_module: ModuleType,
    initial_prompt: str | None = None,
) -> str:
    """音声データを文字起こしする.

    Args:
        audio: モノラルfloat32のnumpy配列
        language: 言語コード（例: "en"）
        mlx_whisper_module: mlx_whisperモジュール（テスト時にモック可能）
        initial_prompt: Whisperへのコンテキストヒント（前回の結果など）

    Returns:
        文字起こしされたテキスト
    """
    result = mlx_whisper_module.transcribe(
        audio,
        path_or_hf_repo=MODEL_REPO,
        language=language,
        initial_prompt=initial_prompt,
        without_timestamps=True,
    )
    text = result["text"]
    # 繰り返しパターンが含まれていれば除去してから返す
    if _has_repetition(text):
        text = _clean_repetition(text)
    return text
