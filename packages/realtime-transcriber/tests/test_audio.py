"""AudioCaptureクラスとfind_device関数とis_silent関数のテスト."""

import queue
from unittest.mock import MagicMock

import numpy as np
import pytest

from realtime_transcriber.audio import (
    SILENCE_RMS_THRESHOLD,
    AudioCapture,
    find_device,
    is_silent,
)


class TestFindDevice:
    """find_device関数のテスト."""

    def test_should_return_device_index_when_exact_name_matches(self) -> None:
        # Given: BlackHole 2chが登録されたデバイスリスト
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Microphone", "max_input_channels": 2},
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]

        # When: BlackHole 2chを検索する
        index = find_device("BlackHole 2ch", mock_sd)

        # Then: 正しいインデックスが返る
        assert index == 1

    def test_should_return_device_index_when_partial_name_matches(self) -> None:
        # Given: 名前に部分一致するデバイス
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Microphone", "max_input_channels": 2},
            {"name": "BlackHole 2ch (Virtual)", "max_input_channels": 2},
        ]

        # When: 部分一致で検索する
        index = find_device("BlackHole 2ch", mock_sd)

        # Then: 部分一致したデバイスのインデックスが返る
        assert index == 1

    def test_should_raise_error_when_device_not_found(self) -> None:
        # Given: 対象デバイスが存在しないリスト
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Microphone", "max_input_channels": 2},
        ]

        # When/Then: 存在しないデバイスを検索するとエラー
        with pytest.raises(RuntimeError):
            find_device("BlackHole 2ch", mock_sd)

    def test_should_raise_error_when_no_devices_available(self) -> None:
        # Given: デバイスリストが空
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = []

        # When/Then: デバイス検索でエラー
        with pytest.raises(RuntimeError):
            find_device("BlackHole 2ch", mock_sd)

    def test_should_only_match_input_capable_devices(self) -> None:
        # Given: 入力チャンネルが0のデバイス（出力専用）
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 0},
            {"name": "BlackHole 2ch Input", "max_input_channels": 2},
        ]

        # When: 入力デバイスとして検索する
        index = find_device("BlackHole 2ch", mock_sd)

        # Then: 入力チャンネルを持つデバイスが返る
        assert index == 1


class TestIsSilent:
    """is_silent関数のテスト."""

    def test_should_return_true_for_all_zeros(self) -> None:
        # Given: 全てゼロの音声データ（完全な無音）
        audio = np.zeros(16000, dtype=np.float32)

        # When: 無音判定する
        result = is_silent(audio, SILENCE_RMS_THRESHOLD)

        # Then: 無音と判定される
        assert result is True

    def test_should_return_false_for_loud_audio(self) -> None:
        # Given: 大きな振幅の音声データ
        audio = np.full(16000, 0.5, dtype=np.float32)

        # When: 無音判定する
        result = is_silent(audio, SILENCE_RMS_THRESHOLD)

        # Then: 無音ではないと判定される
        assert result is False

    def test_should_return_true_when_rms_equals_threshold(self) -> None:
        # Given: RMSがちょうど閾値と等しい音声データ
        # RMS = sqrt(mean(audio^2)) = threshold のとき、全サンプルが同じ値なら
        # audio値 = threshold
        audio = np.full(16000, SILENCE_RMS_THRESHOLD, dtype=np.float32)

        # When: 無音判定する
        result = is_silent(audio, SILENCE_RMS_THRESHOLD)

        # Then: 閾値以下なので無音と判定される
        assert result is True

    def test_should_return_false_when_rms_just_above_threshold(self) -> None:
        # Given: RMSが閾値をわずかに超える音声データ
        value = SILENCE_RMS_THRESHOLD * 1.1
        audio = np.full(16000, value, dtype=np.float32)

        # When: 無音判定する
        result = is_silent(audio, SILENCE_RMS_THRESHOLD)

        # Then: 閾値を超えているので無音ではない
        assert result is False

    def test_should_handle_negative_amplitude_audio(self) -> None:
        # Given: 負の振幅を持つ音声データ（二乗するため符号は無関係）
        audio = np.full(16000, -0.5, dtype=np.float32)

        # When: 無音判定する
        result = is_silent(audio, SILENCE_RMS_THRESHOLD)

        # Then: RMS=0.5 なので無音ではない
        assert result is False

    def test_should_use_rms_calculation(self) -> None:
        # Given: 正弦波の音声データ（RMS = amplitude / sqrt(2)）
        # amplitude=0.02 のとき RMS ≈ 0.01414 > 0.01
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.02

        # When: 無音判定する
        result = is_silent(audio, SILENCE_RMS_THRESHOLD)

        # Then: RMS ≈ 0.014 > 0.01 なので無音ではない
        assert result is False

    def test_should_detect_very_quiet_audio_as_silent(self) -> None:
        # Given: 閾値より十分小さいRMSの音声データ
        audio = np.full(16000, 0.001, dtype=np.float32)

        # When: 無音判定する
        result = is_silent(audio, SILENCE_RMS_THRESHOLD)

        # Then: 無音と判定される
        assert result is True

    def test_should_accept_custom_threshold(self) -> None:
        # Given: カスタム閾値（高め）を指定
        audio = np.full(16000, 0.05, dtype=np.float32)
        high_threshold = 0.1

        # When: カスタム閾値で無音判定する
        result = is_silent(audio, high_threshold)

        # Then: RMS=0.05 < 0.1 なので無音と判定される
        assert result is True


class TestSilenceRmsThreshold:
    """SILENCE_RMS_THRESHOLD定数のテスト."""

    def test_should_be_positive_float(self) -> None:
        # Given/When: 定数の型と値を確認する

        # Then: 正のfloat値である
        assert isinstance(SILENCE_RMS_THRESHOLD, float)
        assert SILENCE_RMS_THRESHOLD > 0


class TestAudioCaptureInit:
    """AudioCaptureのコンストラクタのテスト."""

    def test_should_store_device_name_and_sample_rate(self) -> None:
        # Given: デバイス名とサンプルレート
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]

        # When: AudioCaptureを生成する
        capture = AudioCapture(
            device_name="BlackHole 2ch",
            sample_rate=16000,
            buffer_seconds=3,
            sd_module=mock_sd,
        )

        # Then: パラメータが保持される
        assert capture.sample_rate == 16000
        assert capture.buffer_seconds == 3


class TestAudioCaptureContextManager:
    """AudioCaptureのコンテキストマネージャのテスト."""

    def test_should_open_stream_on_enter(self) -> None:
        # Given: モック化されたsounddevice
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        capture = AudioCapture(
            device_name="BlackHole 2ch",
            sample_rate=16000,
            buffer_seconds=3,
            sd_module=mock_sd,
        )

        # When: コンテキストマネージャに入る
        capture.__enter__()

        # Then: InputStreamが作成・開始される
        mock_sd.InputStream.assert_called_once()
        mock_stream.start.assert_called_once()

    def test_should_close_stream_on_exit(self) -> None:
        # Given: ストリームが開始済みのAudioCapture
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        capture = AudioCapture(
            device_name="BlackHole 2ch",
            sample_rate=16000,
            buffer_seconds=3,
            sd_module=mock_sd,
        )
        capture.__enter__()

        # When: コンテキストマネージャを抜ける
        capture.__exit__(None, None, None)

        # Then: ストリームが停止・クローズされる
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()


class TestAudioCaptureGetAudioChunk:
    """get_audio_chunk メソッドのテスト."""

    def test_should_return_none_when_buffer_insufficient(self) -> None:
        # Given: バッファにデータが不足しているAudioCapture
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        capture = AudioCapture(
            device_name="BlackHole 2ch",
            sample_rate=16000,
            buffer_seconds=3,
            sd_module=mock_sd,
        )

        # When: 音声チャンクを取得する
        chunk = capture.get_audio_chunk()

        # Then: データ不足のためNoneが返る
        assert chunk is None

    def test_should_return_mono_float32_array_when_buffer_sufficient(self) -> None:
        # Given: 3秒分のステレオ音声データがキューに蓄積されている
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        capture = AudioCapture(
            device_name="BlackHole 2ch",
            sample_rate=16000,
            buffer_seconds=3,
            sd_module=mock_sd,
        )
        # キューに十分なステレオデータを入れる（16000Hz * 3秒 = 48000サンプル）
        stereo_data = np.random.rand(48000, 2).astype(np.float32)
        capture._queue = queue.Queue()
        capture._queue.put(stereo_data)

        # When: 音声チャンクを取得する
        chunk = capture.get_audio_chunk()

        # Then: モノラルのfloat32配列が返る
        assert chunk is not None
        assert chunk.ndim == 1
        assert chunk.dtype == np.float32

    def test_should_downmix_stereo_to_mono(self) -> None:
        # Given: 左=1.0, 右=0.0 のステレオ音声
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        capture = AudioCapture(
            device_name="BlackHole 2ch",
            sample_rate=16000,
            buffer_seconds=1,
            sd_module=mock_sd,
        )
        stereo_data = np.zeros((16000, 2), dtype=np.float32)
        stereo_data[:, 0] = 1.0  # 左チャンネルのみ
        stereo_data[:, 1] = 0.0  # 右チャンネルは無音
        capture._queue = queue.Queue()
        capture._queue.put(stereo_data)

        # When: 音声チャンクを取得する
        chunk = capture.get_audio_chunk()

        # Then: モノラルに変換され、平均値（0.5）になる
        assert chunk is not None
        np.testing.assert_allclose(chunk, np.full(16000, 0.5, dtype=np.float32))

    def test_should_concatenate_multiple_queued_chunks(self) -> None:
        # Given: 複数の小さなチャンクがキューに入っている
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        capture = AudioCapture(
            device_name="BlackHole 2ch",
            sample_rate=16000,
            buffer_seconds=2,
            sd_module=mock_sd,
        )
        capture._queue = queue.Queue()
        # 1秒分を2回に分けてキューに入れる（合計2秒）
        chunk1 = np.ones((16000, 2), dtype=np.float32)
        chunk2 = np.ones((16000, 2), dtype=np.float32)
        capture._queue.put(chunk1)
        capture._queue.put(chunk2)

        # When: 音声チャンクを取得する
        chunk = capture.get_audio_chunk()

        # Then: 連結されたモノラル配列が返る（2秒分 = 32000サンプル）
        assert chunk is not None
        assert chunk.shape[0] == 32000


    def test_should_preserve_chunk_order_across_calls(self) -> None:
        # Given: 1回目の呼び出しでは不足、2回目で十分なデータが蓄積される
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        capture = AudioCapture(
            device_name="BlackHole 2ch",
            sample_rate=16000,
            buffer_seconds=2,
            sd_module=mock_sd,
        )
        capture._queue = queue.Queue()

        # 1秒分（不足）
        first_chunk = np.ones((16000, 1), dtype=np.float32) * 0.3
        capture._queue.put(first_chunk)

        # When: 1回目はNone
        assert capture.get_audio_chunk() is None

        # 追加1秒分（合計2秒で十分）
        second_chunk = np.ones((16000, 1), dtype=np.float32) * 0.7
        capture._queue.put(second_chunk)

        # When: 2回目で取得
        chunk = capture.get_audio_chunk()

        # Then: 時系列順序が保たれる（0.3が先、0.7が後）
        assert chunk is not None
        np.testing.assert_allclose(chunk[:16000], 0.3, atol=1e-6)
        np.testing.assert_allclose(chunk[16000:], 0.7, atol=1e-6)


class TestToMono:
    """_to_monoメソッドのテスト."""

    def test_should_convert_stereo_to_mono_via_mean(self) -> None:
        # Given: ステレオ音声（左=1.0, 右=0.0）
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        capture = AudioCapture(
            device_name="BlackHole 2ch",
            sample_rate=16000,
            buffer_seconds=1,
            sd_module=mock_sd,
        )
        stereo = np.zeros((100, 2), dtype=np.float32)
        stereo[:, 0] = 1.0

        # When: モノラル変換する
        mono = capture._to_mono(stereo)

        # Then: 両チャンネルの平均値になる
        assert mono.ndim == 1
        assert mono.dtype == np.float32
        np.testing.assert_allclose(mono, 0.5)

    def test_should_pass_through_mono_input(self) -> None:
        # Given: モノラル音声
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        capture = AudioCapture(
            device_name="BlackHole 2ch",
            sample_rate=16000,
            buffer_seconds=1,
            sd_module=mock_sd,
        )
        mono_input = np.ones(100, dtype=np.float32) * 0.5

        # When: モノラル変換する（既にモノラル）
        result = capture._to_mono(mono_input)

        # Then: そのまま返る
        assert result.ndim == 1
        np.testing.assert_allclose(result, 0.5)


class TestAudioCaptureCallback:
    """InputStreamコールバックの振る舞いテスト."""

    def test_should_copy_indata_to_queue(self) -> None:
        # Given: AudioCaptureのコールバック
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        capture = AudioCapture(
            device_name="BlackHole 2ch",
            sample_rate=16000,
            buffer_seconds=3,
            sd_module=mock_sd,
        )
        capture._queue = queue.Queue()
        indata = np.ones((1024, 2), dtype=np.float32)

        # When: コールバックが呼ばれる
        capture._audio_callback(indata, frames=1024, time_info=None, status=None)

        # Then: キューにコピーされたデータが入る
        queued = capture._queue.get_nowait()
        np.testing.assert_array_equal(queued, indata)
        # コピーであることを確認（元データと同一オブジェクトでない）
        assert queued is not indata
