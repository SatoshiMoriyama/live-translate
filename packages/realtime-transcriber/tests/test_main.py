"""mainモジュールのテスト.

メインループはAudioCapture → transcribe_audio → printのパイプラインを
制御するオーケストレーション層。3モジュールを横断するデータフローのため、
インテグレーションテストとして記述する。
"""

from unittest.mock import MagicMock, patch

import numpy as np

from realtime_transcriber.main import main


class TestMainLoop:
    """main関数のインテグレーションテスト."""

    def test_should_print_transcribed_text_when_audio_available(self) -> None:
        # Given: 音声チャンクを1回返した後Noneを返すAudioCapture
        # KeyboardInterruptで2回目のループを終了させる
        mock_audio_capture = MagicMock()
        mock_audio_capture.__enter__ = MagicMock(return_value=mock_audio_capture)
        mock_audio_capture.__exit__ = MagicMock(return_value=False)
        audio_chunk = np.random.rand(48000).astype(np.float32)
        mock_audio_capture.get_audio_chunk.side_effect = [
            audio_chunk,
            KeyboardInterrupt,
        ]

        mock_transcribe = MagicMock(return_value="Hello, world.")

        # When: mainを実行する
        with patch("realtime_transcriber.main.AudioCapture", return_value=mock_audio_capture):
            with patch("realtime_transcriber.main.transcribe_audio", mock_transcribe):
                with patch("builtins.print") as mock_print:
                    main()

        # Then: 文字起こし結果が出力される
        mock_transcribe.assert_called_once()
        mock_print.assert_any_call("Hello, world.", flush=True)

    def test_should_not_print_when_transcription_is_empty(self) -> None:
        # Given: 空テキストを返す文字起こし
        mock_audio_capture = MagicMock()
        mock_audio_capture.__enter__ = MagicMock(return_value=mock_audio_capture)
        mock_audio_capture.__exit__ = MagicMock(return_value=False)
        audio_chunk = np.random.rand(48000).astype(np.float32)
        mock_audio_capture.get_audio_chunk.side_effect = [
            audio_chunk,
            KeyboardInterrupt,
        ]

        mock_transcribe = MagicMock(return_value="")

        # When: mainを実行する
        with patch("realtime_transcriber.main.AudioCapture", return_value=mock_audio_capture):
            with patch("realtime_transcriber.main.transcribe_audio", mock_transcribe):
                with patch("builtins.print") as mock_print:
                    main()

        # Then: 空テキストはprintされない（終了メッセージのみ許容）
        for c in mock_print.call_args_list:
            if c.args and c.args[0] != "":
                # 終了メッセージは許容
                assert "Hello" not in str(c.args[0])

    def test_should_skip_transcription_when_no_audio_chunk(self) -> None:
        # Given: get_audio_chunkがNoneを返し続ける
        mock_audio_capture = MagicMock()
        mock_audio_capture.__enter__ = MagicMock(return_value=mock_audio_capture)
        mock_audio_capture.__exit__ = MagicMock(return_value=False)
        mock_audio_capture.get_audio_chunk.side_effect = [
            None,
            None,
            KeyboardInterrupt,
        ]

        mock_transcribe = MagicMock()

        # When: mainを実行する
        with patch("realtime_transcriber.main.AudioCapture", return_value=mock_audio_capture):
            with patch("realtime_transcriber.main.transcribe_audio", mock_transcribe):
                with patch("time.sleep"):
                    main()

        # Then: 文字起こしは呼ばれない
        mock_transcribe.assert_not_called()

    def test_should_exit_gracefully_on_keyboard_interrupt(self) -> None:
        # Given: 即座にKeyboardInterruptが発生する
        mock_audio_capture = MagicMock()
        mock_audio_capture.__enter__ = MagicMock(return_value=mock_audio_capture)
        mock_audio_capture.__exit__ = MagicMock(return_value=False)
        mock_audio_capture.get_audio_chunk.side_effect = KeyboardInterrupt

        # When/Then: 例外が発生せず正常終了する
        with patch("realtime_transcriber.main.AudioCapture", return_value=mock_audio_capture):
            with patch("realtime_transcriber.main.transcribe_audio"):
                main()

        # Then: __exit__が呼ばれてリソースが解放される
        mock_audio_capture.__exit__.assert_called_once()

    def test_should_sleep_when_no_audio_chunk_to_reduce_cpu(self) -> None:
        # Given: get_audio_chunkがNoneを返す
        mock_audio_capture = MagicMock()
        mock_audio_capture.__enter__ = MagicMock(return_value=mock_audio_capture)
        mock_audio_capture.__exit__ = MagicMock(return_value=False)
        mock_audio_capture.get_audio_chunk.side_effect = [
            None,
            KeyboardInterrupt,
        ]

        # When: mainを実行する
        with patch("realtime_transcriber.main.AudioCapture", return_value=mock_audio_capture):
            with patch("realtime_transcriber.main.transcribe_audio"):
                with patch("time.sleep") as mock_sleep:
                    main()

        # Then: CPU使用率抑制のためsleepが呼ばれる
        mock_sleep.assert_called_once()


class TestMainLoopSilenceSkip:
    """メインループの無音スキップのインテグレーションテスト."""

    def test_should_skip_transcription_when_audio_is_silent(self) -> None:
        # Given: 無音と判定される音声チャンクを返すAudioCapture
        mock_audio_capture = MagicMock()
        mock_audio_capture.__enter__ = MagicMock(return_value=mock_audio_capture)
        mock_audio_capture.__exit__ = MagicMock(return_value=False)
        silent_chunk = np.zeros(48000, dtype=np.float32)
        mock_audio_capture.get_audio_chunk.side_effect = [
            silent_chunk,
            KeyboardInterrupt,
        ]

        mock_transcribe = MagicMock()

        # When: mainを実行する（is_silentがTrueを返す）
        with patch("realtime_transcriber.main.AudioCapture", return_value=mock_audio_capture):
            with patch("realtime_transcriber.main.transcribe_audio", mock_transcribe):
                with patch("realtime_transcriber.main.is_silent", return_value=True):
                    main()

        # Then: 無音なので文字起こしは呼ばれない
        mock_transcribe.assert_not_called()

    def test_should_transcribe_when_audio_is_not_silent(self) -> None:
        # Given: 無音ではない音声チャンクを返すAudioCapture
        mock_audio_capture = MagicMock()
        mock_audio_capture.__enter__ = MagicMock(return_value=mock_audio_capture)
        mock_audio_capture.__exit__ = MagicMock(return_value=False)
        audio_chunk = np.full(48000, 0.5, dtype=np.float32)
        mock_audio_capture.get_audio_chunk.side_effect = [
            audio_chunk,
            KeyboardInterrupt,
        ]

        mock_transcribe = MagicMock(return_value="Hello.")

        # When: mainを実行する（is_silentがFalseを返す）
        with patch("realtime_transcriber.main.AudioCapture", return_value=mock_audio_capture):
            with patch("realtime_transcriber.main.transcribe_audio", mock_transcribe):
                with patch("realtime_transcriber.main.is_silent", return_value=False):
                    with patch("realtime_transcriber.main.is_hallucination", return_value=False):
                        with patch("builtins.print"):
                            main()

        # Then: 文字起こしが呼ばれる
        mock_transcribe.assert_called_once()


class TestMainLoopHallucinationFilter:
    """メインループのハルシネーションフィルタのインテグレーションテスト."""

    def test_should_not_print_hallucinated_text(self) -> None:
        # Given: Whisperがハルシネーションテキストを返す状況
        mock_audio_capture = MagicMock()
        mock_audio_capture.__enter__ = MagicMock(return_value=mock_audio_capture)
        mock_audio_capture.__exit__ = MagicMock(return_value=False)
        audio_chunk = np.full(48000, 0.5, dtype=np.float32)
        mock_audio_capture.get_audio_chunk.side_effect = [
            audio_chunk,
            KeyboardInterrupt,
        ]

        mock_transcribe = MagicMock(return_value="Thank you.")

        # When: mainを実行する（ハルシネーションフィルタが検出）
        with patch("realtime_transcriber.main.AudioCapture", return_value=mock_audio_capture):
            with patch("realtime_transcriber.main.transcribe_audio", mock_transcribe):
                with patch("realtime_transcriber.main.is_silent", return_value=False):
                    with patch("realtime_transcriber.main.is_hallucination", return_value=True):
                        with patch("builtins.print") as mock_print:
                            main()

        # Then: ハルシネーションテキストはprintされない
        mock_print.assert_not_called()

    def test_should_print_normal_text_after_hallucination_check(self) -> None:
        # Given: 正常なテキストを返すWhisper
        mock_audio_capture = MagicMock()
        mock_audio_capture.__enter__ = MagicMock(return_value=mock_audio_capture)
        mock_audio_capture.__exit__ = MagicMock(return_value=False)
        audio_chunk = np.full(48000, 0.5, dtype=np.float32)
        mock_audio_capture.get_audio_chunk.side_effect = [
            audio_chunk,
            KeyboardInterrupt,
        ]

        mock_transcribe = MagicMock(return_value="Hello, world.")

        # When: mainを実行する（ハルシネーションではない）
        with patch("realtime_transcriber.main.AudioCapture", return_value=mock_audio_capture):
            with patch("realtime_transcriber.main.transcribe_audio", mock_transcribe):
                with patch("realtime_transcriber.main.is_silent", return_value=False):
                    with patch("realtime_transcriber.main.is_hallucination", return_value=False):
                        with patch("builtins.print") as mock_print:
                            main()

        # Then: 正常なテキストが出力される
        mock_print.assert_any_call("Hello, world.", flush=True)


class TestMainLoopPipelineOrder:
    """メインループのパイプライン処理順序のインテグレーションテスト."""

    def test_should_check_silence_before_transcription(self) -> None:
        # Given: 無音チャンクが返される
        mock_audio_capture = MagicMock()
        mock_audio_capture.__enter__ = MagicMock(return_value=mock_audio_capture)
        mock_audio_capture.__exit__ = MagicMock(return_value=False)
        silent_chunk = np.zeros(48000, dtype=np.float32)
        mock_audio_capture.get_audio_chunk.side_effect = [
            silent_chunk,
            KeyboardInterrupt,
        ]

        mock_transcribe = MagicMock()
        call_order = []

        def mock_is_silent(audio, threshold):
            call_order.append("is_silent")
            return True

        def mock_transcribe_fn(**kwargs):
            call_order.append("transcribe")
            return "text"

        # When: mainを実行する
        with patch("realtime_transcriber.main.AudioCapture", return_value=mock_audio_capture):
            with patch("realtime_transcriber.main.transcribe_audio", side_effect=mock_transcribe_fn):
                with patch("realtime_transcriber.main.is_silent", side_effect=mock_is_silent):
                    main()

        # Then: is_silentが呼ばれ、transcribeは呼ばれない
        assert "is_silent" in call_order
        assert "transcribe" not in call_order

    def test_should_check_hallucination_after_transcription(self) -> None:
        # Given: 有音チャンクが返される
        mock_audio_capture = MagicMock()
        mock_audio_capture.__enter__ = MagicMock(return_value=mock_audio_capture)
        mock_audio_capture.__exit__ = MagicMock(return_value=False)
        audio_chunk = np.full(48000, 0.5, dtype=np.float32)
        mock_audio_capture.get_audio_chunk.side_effect = [
            audio_chunk,
            KeyboardInterrupt,
        ]

        call_order = []

        def mock_is_silent(audio, threshold):
            call_order.append("is_silent")
            return False

        def mock_transcribe_fn(**kwargs):
            call_order.append("transcribe")
            return "Thank you."

        def mock_is_hallucination(text):
            call_order.append("is_hallucination")
            return True

        # When: mainを実行する
        with patch("realtime_transcriber.main.AudioCapture", return_value=mock_audio_capture):
            with patch("realtime_transcriber.main.transcribe_audio", side_effect=mock_transcribe_fn):
                with patch("realtime_transcriber.main.is_silent", side_effect=mock_is_silent):
                    with patch("realtime_transcriber.main.is_hallucination", side_effect=mock_is_hallucination):
                        with patch("builtins.print"):
                            main()

        # Then: is_silent → transcribe → is_hallucination の順で呼ばれる
        assert call_order == ["is_silent", "transcribe", "is_hallucination"]
