"""音声キャプチャモジュール.

sounddeviceによるBlackHole 2chからの音声取得と前処理を担当する。
VADによる発話区間検出で自然な文の区切りで音声を返す。
"""

import queue
from collections import deque
from types import ModuleType, TracebackType

import numpy as np
from silero_vad_lite import SileroVAD

# VAD設定
VAD_THRESHOLD = 0.5
MIN_SILENCE_MS = 800
MAX_SPEECH_SECONDS = 30
MIN_SPEECH_SECONDS = 1


def find_device(name: str, sd_module: ModuleType) -> int:
    """デバイス名で入力デバイスを検索し、インデックスを返す."""
    devices = sd_module.query_devices()
    for index, device in enumerate(devices):
        if name in device["name"] and device["max_input_channels"] > 0:
            return index

    available = [d["name"] for d in devices]
    raise RuntimeError(f"Device '{name}' not found. Available devices: {available}")


class AudioCapture:
    """VADベースの音声キャプチャ.

    発話開始を検出してバッファリングし、無音区間で区切って返す。
    """

    def __init__(
        self,
        device_name: str,
        sample_rate: int,
        sd_module: ModuleType,
    ) -> None:
        self._sd = sd_module
        self._device_index = find_device(device_name, sd_module)
        self.sample_rate = sample_rate
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None

        # VAD
        self._vad = SileroVAD(sample_rate)
        self._window_size = self._vad.window_size_samples

        # 発話状態管理
        self._in_speech = False
        self._speech_chunks: list[np.ndarray] = []
        self._silence_samples = 0
        self._speech_samples = 0
        self._min_silence_samples = int(sample_rate * MIN_SILENCE_MS / 1000)
        self._max_speech_samples = int(sample_rate * MAX_SPEECH_SECONDS)
        self._min_speech_samples = int(sample_rate * MIN_SPEECH_SECONDS)
        self._last_status = ""

        # モノラル変換用の未処理バッファ（dequeで蓄積し処理時に結合）
        self._mono_chunks: deque[np.ndarray] = deque()
        self._mono_buffer = np.array([], dtype=np.float32)

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
        """sounddevice InputStreamコールバック."""
        self._queue.put(indata.copy())

    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        """ステレオ音声をモノラルに変換する."""
        if audio.ndim == 2:
            return np.mean(audio, axis=1).astype(np.float32)
        return audio.astype(np.float32)

    def get_audio_chunk(self) -> np.ndarray | None:
        """VADで発話区間を検出し、発話終了時にまとめて返す.

        Returns:
            発話区間のモノラルfloat32配列、またはまだ発話中/無音時にNone
        """
        # キューからデータを取り出してモノラルに変換（dequeに蓄積）
        while True:
            try:
                stereo = self._queue.get_nowait()
            except queue.Empty:
                break
            self._mono_chunks.append(self._to_mono(stereo))

        # 蓄積したチャンクをまとめてバッファに結合
        if self._mono_chunks:
            self._mono_chunks.appendleft(self._mono_buffer)
            self._mono_buffer = np.concatenate(self._mono_chunks)
            self._mono_chunks.clear()

        # VADウィンドウサイズ単位で処理
        result = None
        while len(self._mono_buffer) >= self._window_size:
            window = self._mono_buffer[: self._window_size]
            self._mono_buffer = self._mono_buffer[self._window_size :]

            prob = self._vad.process(window.tobytes())
            is_speech = prob >= VAD_THRESHOLD

            if is_speech:
                if not self._in_speech:
                    self._in_speech = True
                    self._silence_samples = 0
                    self._speech_samples = 0
                self._speech_chunks.append(window)
                self._speech_samples += self._window_size
                self._silence_samples = 0

                # 秒単位で変わった時だけ更新
                elapsed = int(self._speech_samples / self.sample_rate)
                status = f"\r\033[93m● Recording... {elapsed}s\033[0m"
                if status != self._last_status:
                    print(status, end="", flush=True)
                    self._last_status = status
            elif self._in_speech:
                # 発話中の無音
                self._speech_chunks.append(window)
                self._silence_samples += self._window_size
                self._speech_samples += self._window_size

                # 無音が十分続いたら発話終了
                if self._silence_samples >= self._min_silence_samples:
                    result = self._finalize_speech()
                    break

            # 最大長に達したら強制区切り
            if self._in_speech and self._speech_samples >= self._max_speech_samples:
                result = self._finalize_speech()
                break

        return result

    def _finalize_speech(self) -> np.ndarray | None:
        """蓄積した発話チャンクを結合して返す."""
        print("\r\033[K", end="", flush=True)  # 録音中表示をクリア
        self._last_status = ""

        if not self._speech_chunks:
            return None

        audio = np.concatenate(self._speech_chunks)

        # 末尾の無音部分をトリムしてWhisperの処理を効率化
        if self._silence_samples > 0 and self._silence_samples < len(audio):
            audio = audio[: -self._silence_samples]

        self._speech_chunks.clear()
        self._in_speech = False
        self._silence_samples = 0
        self._speech_samples = 0

        # 短すぎる発話は無視
        if len(audio) < self._min_speech_samples:
            return None

        return audio
