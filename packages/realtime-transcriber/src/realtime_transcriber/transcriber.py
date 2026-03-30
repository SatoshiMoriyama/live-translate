"""文字起こしモジュール.

mlx-whisperによる音声テキスト変換を担当する。
"""

import re
from types import ModuleType

import numpy as np

# Apple Silicon向けに最適化されたWhisperモデル
MODEL_REPO = "mlx-community/whisper-large-v3-turbo"

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
    match = re.search(r"\b(\w+)(?:\s+\1){" + str(min_repeats - 1) + r",}", text.lower())
    return match is not None


def _clean_repetition(text: str) -> str:
    """テキストから異常な繰り返し部分を除去する."""
    # 同じ単語の5回以上の連続を1回に置換
    cleaned = re.sub(r"\b(\w+)(?:\s+\1){4,}", r"\1", text)
    return cleaned.strip()


def is_hallucination(text: str) -> bool:
    """Whisperの出力テキストが既知のハルシネーションパターンに一致するか判定する."""
    normalized = text.strip().lower()
    if not normalized:
        return False
    if normalized in HALLUCINATION_PATTERNS:
        return True
    # 繰り返しが大半を占める場合はハルシネーション
    # （例: "too too too too too too" → 除去後が元の30%未満ならハルシネーション）
    if _has_repetition(normalized):
        cleaned = _clean_repetition(normalized)
        if len(cleaned) < len(normalized) * 0.3:
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
    )
    return result["text"]
