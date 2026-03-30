"""音声キャプチャモジュール.

sounddeviceによるBlackHole 2chからの音声取得と前処理を担当する。
"""

import queue
from types import ModuleType, TracebackType

import numpy as np

SILENCE_RMS_THRESHOLD = 0.0005


def is_silent(audio: np.ndarray, threshold: float = SILENCE_RMS_THRESHOLD) -> bool:
    """音声データのRMSレベルが閾値以下かどうかを判定する.

    Args:
        audio: モノラルfloat32のnumpy配列
        threshold: RMS閾値（この値以下なら無音と判定）

    Returns:
        無音ならTrue
    """
    rms = float(np.sqrt(np.mean(audio**2)))
    return rms <= threshold


def find_device(name: str, sd_module: ModuleType) -> int:
    """デバイス名で入力デバイスを検索し、インデックスを返す.

    部分一致で検索し、入力チャンネルを持つデバイスのみ対象とする。

    Args:
        name: 検索するデバイス名（部分一致）
        sd_module: sounddeviceモジュール（テスト時にモック可能）

    Returns:
        デバイスインデックス

    Raises:
        RuntimeError: デバイスが見つからない場合
    """
    devices = sd_module.query_devices()
    for index, device in enumerate(devices):
        if name in device["name"] and device["max_input_channels"] > 0:
            return index

    available = [d["name"] for d in devices]
    raise RuntimeError(f"Device '{name}' not found. Available devices: {available}")


class AudioCapture:
    """sounddeviceを使った音声キャプチャ.

    コンテキストマネージャとして使用し、ストリームのライフサイクルを管理する。
    """

    def __init__(
        self,
        device_name: str,
        sample_rate: int,
        buffer_seconds: int,
        sd_module: ModuleType,
    ) -> None:
        self._sd = sd_module
        self._device_index = find_device(device_name, sd_module)
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self._required_samples = sample_rate * buffer_seconds
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._pending_chunks: list[np.ndarray] = []
        self._stream = None

    def __enter__(self) -> "AudioCapture":
        self._stream = self._sd.InputStream(
            device=self._device_index,
            samplerate=self.sample_rate,
            channels=2,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """sounddevice InputStreamコールバック. データをコピーしてキューに追加する."""
        self._queue.put(indata.copy())

    def get_audio_chunk(self) -> np.ndarray | None:
        """キューから音声を取り出し、モノラルfloat32配列として返す.

        バッファが必要サンプル数に達していなければNoneを返す。

        Returns:
            モノラルfloat32のnumpy配列、またはデータ不足時にNone
        """
        while not self._queue.empty():
            self._pending_chunks.append(self._queue.get_nowait())

        total_samples = sum(c.shape[0] for c in self._pending_chunks)
        if total_samples < self._required_samples:
            return None

        concatenated = np.concatenate(self._pending_chunks, axis=0)
        self._pending_chunks.clear()

        return self._to_mono(concatenated)

    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        """ステレオ音声をモノラルに変換する."""
        if audio.ndim == 2:
            return np.mean(audio, axis=1).astype(np.float32)
        return audio.astype(np.float32)
