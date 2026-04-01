"""翻訳モジュール.

Amazon BedrockまたはAWS Translateによる英日翻訳を担当する。
TRANSLATION_BACKEND で切り替え可能。デフォルトは "bedrock"。
Bedrockの場合、BEDROCK_MODEL_ID でモデルを指定できる。
"""

import logging

import boto3
import botocore.client

logger = logging.getLogger(__name__)

# --- 翻訳設定 ---
# 翻訳バックエンド: "bedrock" または "aws_translate"
TRANSLATION_BACKEND = "bedrock"

# Bedrock設定
BEDROCK_REGION = "us-east-1"
# 使用するBedrockモデル（クロスリージョン推論プロファイル）
# - Amazon Nova Pro:   "us.amazon.nova-pro-v1:0"（デフォルト、高品質・高速・低コスト）
# - Claude Haiku 4.5:  "us.anthropic.claude-haiku-4-5-20251001-v1:0"（高品質、Nova Proより高価）
# - Amazon Nova Lite:  "us.amazon.nova-lite-v1:0"（高速、低コスト、品質はやや劣る）
# - Amazon Nova Micro: "us.amazon.nova-micro-v1:0"（最速、最安、短文向き）
BEDROCK_MODEL_ID = "us.amazon.nova-pro-v1:0"

# AWS Translate設定
AWS_TRANSLATE_REGION = "ap-northeast-1"


def create_translate_client() -> botocore.client.BaseClient:
    """翻訳クライアントを生成する.

    TRANSLATION_BACKEND に応じて Bedrock または AWS Translate のクライアントを返す。
    """
    if TRANSLATION_BACKEND == "bedrock":
        return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    else:
        return boto3.client("translate", region_name=AWS_TRANSLATE_REGION)


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    client: botocore.client.BaseClient,
    session_context: str = "",
) -> str:
    """テキストを翻訳する.

    TRANSLATION_BACKEND に応じて Bedrock または AWS Translate を使用する。

    Args:
        text: 翻訳元テキスト
        source_lang: ソース言語コード（例: "en"）
        target_lang: ターゲット言語コード（例: "ja"）
        client: 翻訳クライアント
        session_context: セッションの要約（Bedrock翻訳の文脈として使用）

    Returns:
        翻訳されたテキスト
    """
    if TRANSLATION_BACKEND == "bedrock":
        return _translate_with_bedrock(
            text, source_lang, target_lang, client, session_context
        )
    else:
        return _translate_with_aws_translate(text, source_lang, target_lang, client)


def _translate_with_bedrock(
    text: str,
    source_lang: str,
    target_lang: str,
    client: botocore.client.BaseClient,
    session_context: str = "",
) -> str:
    """Bedrockでテキストを翻訳する.

    systemフィールドに翻訳指示を分離し、userロールには翻訳対象テキストのみを渡す。
    """
    system_prompt = (
        f"Translate the following {source_lang} text to natural {target_lang}. "
        "Translate so that it is easy for Japanese speakers to understand. "
        "For technical terms that are commonly used in English (e.g. AWS, API), "
        "keep them in English. "
        "Output ONLY the translated text. "
        "Do not add explanations, context, or commentary."
    )
    if session_context:
        system_prompt += (
            f"\n\nSession context for reference: {session_context}"
        )
    response = client.converse(
        modelId=BEDROCK_MODEL_ID,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": text}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.1},
    )
    return response["output"]["message"]["content"][0]["text"].strip()


def _translate_with_aws_translate(
    text: str,
    source_lang: str,
    target_lang: str,
    client: botocore.client.BaseClient,
) -> str:
    """AWS Translateでテキストを翻訳する."""
    response = client.translate_text(
        Text=text,
        SourceLanguageCode=source_lang,
        TargetLanguageCode=target_lang,
    )
    return response["TranslatedText"]
