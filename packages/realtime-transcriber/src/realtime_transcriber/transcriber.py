"""文字起こしモジュール.

mlx-whisperによる音声テキスト変換を担当する。
"""

from types import ModuleType

import numpy as np

MODEL_REPO = "mlx-community/whisper-large-v3-turbo"

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


def is_hallucination(text: str) -> bool:
    """Whisperの出力テキストが既知のハルシネーションパターンに一致するか判定する.

    Args:
        text: Whisperの出力テキスト

    Returns:
        ハルシネーションならTrue
    """
    normalized = text.strip().lower()
    if not normalized:
        return False
    return normalized in HALLUCINATION_PATTERNS


def transcribe_audio(
    audio: np.ndarray,
    language: str,
    mlx_whisper_module: ModuleType,
) -> str:
    """音声データを文字起こしする.

    Args:
        audio: モノラルfloat32のnumpy配列
        language: 言語コード（例: "en"）
        mlx_whisper_module: mlx_whisperモジュール（テスト時にモック可能）

    Returns:
        文字起こしされたテキスト
    """
    result = mlx_whisper_module.transcribe(
        audio,
        path_or_hf_repo=MODEL_REPO,
        language=language,
    )
    return result["text"]
