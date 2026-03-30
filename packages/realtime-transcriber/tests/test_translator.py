"""translator モジュールのテスト.

Amazon Translate を使った英日翻訳機能をテストする。
"""

from unittest.mock import MagicMock

import pytest

from realtime_transcriber.translator import (
    SOURCE_LANGUAGE,
    TARGET_LANGUAGE,
    translate_text,
)


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
        result = translate_text("Hello, world.", mock_client)

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
        translate_text("test", mock_client)

        # Then: 正しいパラメータでAPIが呼ばれる
        mock_client.translate_text.assert_called_once_with(
            Text="test",
            SourceLanguageCode=SOURCE_LANGUAGE,
            TargetLanguageCode=TARGET_LANGUAGE,
        )

    def test_should_return_empty_string_for_empty_input(self) -> None:
        # Given: 空文字列
        mock_client = MagicMock()

        # When: 空文字列を翻訳する
        result = translate_text("", mock_client)

        # Then: 空文字列が返り、APIは呼ばれない
        assert result == ""
        mock_client.translate_text.assert_not_called()

    def test_should_propagate_api_error(self) -> None:
        # Given: エラーを返すTranslateクライアント
        mock_client = MagicMock()
        mock_client.translate_text.side_effect = Exception("ServiceUnavailable")

        # When/Then: エラーがそのまま伝播する
        with pytest.raises(Exception, match="ServiceUnavailable"):
            translate_text("Hello", mock_client)

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
        result = translate_text(long_text, mock_client)

        # Then: 翻訳結果が返る
        assert len(result) > 0
        mock_client.translate_text.assert_called_once()


class TestTranslatorConstants:
    """翻訳モジュールの定数テスト."""

    def test_source_language_should_be_english(self) -> None:
        # Given/When: ソース言語定数を確認する

        # Then: 英語である
        assert SOURCE_LANGUAGE == "en"

    def test_target_language_should_be_japanese(self) -> None:
        # Given/When: ターゲット言語定数を確認する

        # Then: 日本語である
        assert TARGET_LANGUAGE == "ja"
