"""要約モジュール.

Amazon Bedrock（Claude Haiku 4.5）を使い、セッションの内容を定期的に要約する。
「前回の要約 + 直近の新テキスト」から更新要約を生成する方式。
"""

import logging
import threading
import time

import boto3

logger = logging.getLogger(__name__)

# Bedrock設定（Haiku 4.5はクロスリージョン推論プロファイルのみ対応）
BEDROCK_REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SUMMARY_INTERVAL_SECONDS = 60


def _build_prompt(prev_summary: str, recent_texts: list[str]) -> str:
    """要約リクエスト用のプロンプトを組み立てる."""
    recent_block = "\n".join(recent_texts)

    if prev_summary:
        return (
            "あなたはリアルタイム翻訳セッションの要約アシスタントです。\n"
            "以下の「これまでの要約」と「直近の発話内容」を踏まえて、"
            "セッション全体の要約を日本語で更新してください。\n"
            "箇条書きで、重要なポイントを簡潔にまとめてください。\n\n"
            f"## これまでの要約\n{prev_summary}\n\n"
            f"## 直近の発話内容\n{recent_block}\n\n"
            "## 更新された要約"
        )
    return (
        "あなたはリアルタイム翻訳セッションの要約アシスタントです。\n"
        "以下の発話内容を日本語で要約してください。\n"
        "箇条書きで、重要なポイントを簡潔にまとめてください。\n\n"
        f"## 発話内容\n{recent_block}\n\n"
        "## 要約"
    )


def _invoke_bedrock(client: object, prompt: str) -> str:
    """Bedrock Converse APIで要約を生成する."""
    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.3},
    )
    return response["output"]["message"]["content"][0]["text"]


class Summarizer:
    """バックグラウンドで定期的に要約を生成するワーカー.

    メインループをブロックしないようにデーモンスレッドで動作する。
    """

    def __init__(self, session_logger: object) -> None:
        self._session_logger = session_logger
        self._client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        self._prev_summary = ""
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """要約ワーカーを開始する."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def latest_summary(self) -> str:
        """最新の要約テキストを返す（スレッドセーフ）."""
        with self._lock:
            return self._prev_summary

    def stop(self) -> None:
        """要約ワーカーを停止する."""
        self._running = False

    def _loop(self) -> None:
        """SUMMARY_INTERVAL_SECONDS ごとに要約を生成するループ."""
        while self._running:
            time.sleep(SUMMARY_INTERVAL_SECONDS)
            if not self._running:
                break
            self._generate_summary()

    def _generate_summary(self) -> None:
        """蓄積テキストから要約を生成し、ターミナルとログに出力する."""
        recent = self._session_logger.flush_recent()
        if not recent:
            return

        prompt = _build_prompt(self._prev_summary, recent)
        try:
            summary = _invoke_bedrock(self._client, prompt)
        except Exception:
            logger.exception("Summary generation failed")
            return

        with self._lock:
            self._prev_summary = summary
        self._session_logger.log_summary(summary)

        # ターミナルに要約を表示
        print("\n\033[96m--- 要約 ---\033[0m", flush=True)
        for line in summary.strip().splitlines():
            print(f"\033[96m{line}\033[0m", flush=True)
        print("\033[96m---\033[0m\n", flush=True)
