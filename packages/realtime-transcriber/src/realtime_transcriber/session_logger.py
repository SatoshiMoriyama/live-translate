"""セッションログモジュール.

起動ごとにログファイルを作成し、文字起こし・翻訳結果をタイムスタンプ付きで記録する。
ログは logs/ ディレクトリに保存される。
"""

from datetime import datetime
from pathlib import Path

# ログ出力先ディレクトリ（プロジェクトルート/logs/）
_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


class SessionLogger:
    """1セッション（1回の起動）に対応するログファイルを管理する."""

    def __init__(self) -> None:
        _LOGS_DIR.mkdir(exist_ok=True)
        self._start_time = datetime.now()
        filename = self._start_time.strftime("%Y-%m-%d_%H%M%S.log")
        self._path = _LOGS_DIR / filename
        # ヘッダーを書き込む
        self._path.write_text(
            f"# Session started at {self._start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            encoding="utf-8",
        )

    @property
    def path(self) -> Path:
        """ログファイルのパスを返す."""
        return self._path

    def _elapsed(self) -> str:
        """セッション開始からの経過時間を [MM:SS] 形式で返す."""
        delta = datetime.now() - self._start_time
        total_seconds = int(delta.total_seconds())
        minutes, seconds = divmod(total_seconds, 60)
        return f"[{minutes:02d}:{seconds:02d}]"

    def log(self, sentence: str, translated: str) -> None:
        """原文と翻訳文を1エントリとして記録する."""
        ts = self._elapsed()
        with self._path.open("a", encoding="utf-8") as f:
            f.write(f"{ts} {sentence}\n")
            f.write(f"{ts} {translated}\n\n")
