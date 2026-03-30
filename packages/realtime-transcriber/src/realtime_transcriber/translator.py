"""翻訳モジュール.

AWS Translateによる英日翻訳を担当する。
"""

import boto3


def create_translate_client(region: str = "ap-northeast-1") -> boto3.client:
    """AWS Translateクライアントを生成する."""
    return boto3.client("translate", region_name=region)


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    client: boto3.client,
) -> str:
    """AWS Translateでテキストを翻訳する.

    Args:
        text: 翻訳元テキスト
        source_lang: ソース言語コード（例: "en"）
        target_lang: ターゲット言語コード（例: "ja"）
        client: AWS Translateクライアント

    Returns:
        翻訳されたテキスト
    """
    response = client.translate_text(
        Text=text,
        SourceLanguageCode=source_lang,
        TargetLanguageCode=target_lang,
    )
    return response["TranslatedText"]
