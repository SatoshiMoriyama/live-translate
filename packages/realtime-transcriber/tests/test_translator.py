"""translator モジュールのテスト.

Amazon Translate を使った翻訳機能をテストする。
"""

from unittest.mock import MagicMock

import pytest

from realtime_transcriber.translator import translate_text


class TestTranslateText:
    """translate_text関数のテスト."""

    def test_should_return_translated_text(self) -> None:
        # Given: 英語テキストと正常に翻訳を返すTranslateクライアント
        mock_client = MagicMock()
        mock_client.translate_text.return_value = {
            "TranslatedText": "こんにちは、世界。",
            "SourceLanguageCode": "en",
            "TargetLanguageCode": "ja",
        }

        # When: 翻訳を実行する
        result = translate_text(
            text="Hello, world.",
            source_lang="en",
            target_lang="ja",
            client=mock_client,
        )

        # Then: 日本語翻訳テキストが返る
        assert result == "こんにちは、世界。"

    def test_should_call_translate_with_correct_parameters(self) -> None:
        # Given: モック化されたTranslateクライアント
        mock_client = MagicMock()
        mock_client.translate_text.return_value = {
            "TranslatedText": "テスト",
            "SourceLanguageCode": "en",
            "TargetLanguageCode": "ja",
        }

        # When: 翻訳を実行する
        translate_text(
            text="test",
            source_lang="en",
            target_lang="ja",
            client=mock_client,
        )

        # Then: 正しいパラメータでAPIが呼ばれる
        mock_client.translate_text.assert_called_once_with(
            Text="test",
            SourceLanguageCode="en",
            TargetLanguageCode="ja",
        )

    def test_should_propagate_api_error(self) -> None:
        # Given: エラーを返すTranslateクライアント
        mock_client = MagicMock()
        mock_client.translate_text.side_effect = Exception("ServiceUnavailable")

        # When/Then: エラーがそのまま伝播する
        with pytest.raises(Exception, match="ServiceUnavailable"):
            translate_text(
                text="Hello",
                source_lang="en",
                target_lang="ja",
                client=mock_client,
            )

    def test_should_handle_long_text(self) -> None:
        # Given: 長いテキスト
        long_text = "This is a very long sentence. " * 100
        mock_client = MagicMock()
        mock_client.translate_text.return_value = {
            "TranslatedText": "これはとても長い文です。" * 100,
            "SourceLanguageCode": "en",
            "TargetLanguageCode": "ja",
        }

        # When: 長いテキストを翻訳する
        result = translate_text(
            text=long_text,
            source_lang="en",
            target_lang="ja",
            client=mock_client,
        )

        # Then: 翻訳結果が返る
        assert len(result) > 0
        mock_client.translate_text.assert_called_once()
