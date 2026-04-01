"""要約モジュール.

Amazon Bedrock（Claude Haiku 4.5）を使い、セッションの内容を定期的に要約する。
「前回の要約 + 直近の新テキスト」から更新要約を生成する方式。
日本語要約（表示・ログ用）と英語キーワード要約（Whisper initial_prompt用）を
1回のAPI呼び出しで同時に生成する。
"""

import json
import logging
import threading

import boto3

from realtime_transcriber.session_logger import SessionLogger
from realtime_transcriber.translator import BEDROCK_REGION

logger = logging.getLogger(__name__)

# 要約用Bedrockモデル（Haiku 4.5はクロスリージョン推論プロファイルのみ対応）
BEDROCK_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SUMMARY_INTERVAL_SECONDS = 60


def _build_prompt(prev_summary_ja: str, recent_texts: list[str]) -> str:
    """要約リクエスト用のプロンプトを組み立てる.

    1回のAPI呼び出しで日本語要約と英語キーワード要約を同時に生成する。
    """
    recent_block = "\n".join(recent_texts)

    context_section = ""
    if prev_summary_ja:
        context_section = f"## これまでの要約\n{prev_summary_ja}\n\n"

    return f"{context_section}## 直近の発話内容\n{recent_block}"


def _invoke_bedrock(client: object, prompt: str) -> str:
    """Bedrock Converse APIで要約を生成する.

    systemフィールドに指示を分離し、userロールには発話内容のみを渡す。
    """
    system_prompt = (
        "あなたはリアルタイム英語セッションを聴いている日本人向けの要約アシスタントです。\n"
        "与えられた発話内容を踏まえて、以下のJSON形式で出力してください。他のテキストは不要です。\n\n"
        "```json\n"
        "{\n"
        '  "summary_ja": "セッション全体の要約を日本語の自然な文章で記述してください。'
        "英語圏特有の表現・固有名詞・略語は日本語話者にわかりやすく補足してください。"
        '3〜5文程度でまとめてください。",\n'
        '  "prompt_en": "English keyword summary of the session topics, '
        "key terms, speaker names, and technical vocabulary. "
        'Under 400 characters. No markdown, plain text only."\n'
        "}\n"
        "```"
    )
    response = client.converse(
        modelId=BEDROCK_MODEL_ID,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 1024, "temperature": 0.3},
    )
    return response["output"]["message"]["content"][0]["text"]


def _parse_response(raw: str) -> tuple[str, str]:
    """Bedrockのレスポンスからsummary_jaとprompt_enを抽出する.

    Returns:
        (summary_ja, prompt_en) のタプル。パース失敗時はrawをsummary_jaとして返す。
    """
    # ```json ... ``` ブロックの中身を抽出
    text = raw.strip()
    if "```" in text:
        # コードブロック内のJSONを抽出
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

    try:
        data = json.loads(text)
        return data.get("summary_ja", ""), data.get("prompt_en", "")
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Failed to parse summary JSON, using raw text")
        return raw, ""


class Summarizer:
    """バックグラウンドで定期的に要約を生成するワーカー.

    メインループをブロックしないようにデーモンスレッドで動作する。
    """

    def __init__(self, session_logger: SessionLogger) -> None:
        self._session_logger = session_logger
        self._client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        self._prev_summary_ja = ""
        self._prompt_en = ""
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """要約ワーカーを開始する."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def latest_summary(self) -> str:
        """最新の日本語要約を返す（スレッドセーフ）."""
        with self._lock:
            return self._prev_summary_ja

    @property
    def whisper_hint(self) -> str:
        """Whisper initial_prompt 用の英語キーワード要約を返す（スレッドセーフ）."""
        with self._lock:
            return self._prompt_en

    def stop(self) -> None:
        """要約ワーカーを即座に停止する."""
        self._stop_event.set()

    def _loop(self) -> None:
        """SUMMARY_INTERVAL_SECONDS ごとに要約を生成するループ."""
        while not self._stop_event.wait(timeout=SUMMARY_INTERVAL_SECONDS):
            self._generate_summary()

    def _generate_summary(self) -> None:
        """蓄積テキストから要約を生成し、ターミナルとログに出力する."""
        recent = self._session_logger.flush_recent()
        if not recent:
            return

        prompt = _build_prompt(self._prev_summary_ja, recent)
        try:
            raw = _invoke_bedrock(self._client, prompt)
        except Exception:
            logger.exception("Summary generation failed")
            return

        summary_ja, prompt_en = _parse_response(raw)

        with self._lock:
            self._prev_summary_ja = summary_ja
            if prompt_en:
                self._prompt_en = prompt_en[:400]
        self._session_logger.log_summary(summary_ja)
        if prompt_en:
            self._session_logger.log_whisper_hint(prompt_en[:400])

        # ターミナルに要約を表示
        print("\n\033[96m--- 要約 ---\033[0m", flush=True)
        for line in summary_ja.strip().splitlines():
            print(f"\033[96m{line}\033[0m", flush=True)
        print("\033[96m---\033[0m\n", flush=True)
