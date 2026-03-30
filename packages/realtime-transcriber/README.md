# realtime-transcriber

macOS のシステム音声をリアルタイムで文字起こし・翻訳する CLI ツールです。

BlackHole 2ch 経由でキャプチャした音声を MLX-Whisper で文字起こしし、AWS Translate で日本語に翻訳して表示します。

## アーキテクチャ

```
システム音声 → BlackHole 2ch → sounddevice
  → VAD（Silero VAD）で発話区間を検出
  → MLX-Whisper で文字起こし
  → AWS Translate で英→日翻訳（複数文は並列処理）
  → ターミナルに表示
```

## 前提条件

- macOS（Apple Silicon）
- Python 3.11〜3.13
- [uv](https://docs.astral.sh/uv/) （パッケージ管理）
- [BlackHole 2ch](https://existential.audio/blackhole/) （仮想オーディオデバイス）
- macOS の「Audio MIDI 設定」で Multi-Output Device を作成済み
- AWS CLI で SSO ログイン済み（`aws sso login`）

## セットアップ

```bash
# 依存パッケージのインストール
uv sync

# AWS SSO ログイン
aws sso login
```

## 使い方

```bash
uv run realtime-transcriber
```

起動すると音声出力先の確認が行われます。Multi-Output Device に切り替えた後、Enter を押すと文字起こしが開始されます。

終了は `Ctrl+C` です。

## 出力例

```
● Recording... 3s
  Hope you all had a great time at re:Invent so far!
  これまでのre:Inventで素晴らしい時間を過ごしたことを願っています!

  Raise your hand if you ever been paged at midnight.
  真夜中にページングされたことがある方は手を挙げてください。
```

## 主要モジュール

| ファイル | 役割 |
|---------|------|
| `main.py` | CLI エントリポイント。パイプライン全体の制御 |
| `audio.py` | 音声キャプチャと VAD による発話区間検出 |
| `transcriber.py` | MLX-Whisper による文字起こしとハルシネーション除去 |
| `translator.py` | AWS Translate による翻訳 |

## テスト

```bash
uv run pytest
```
