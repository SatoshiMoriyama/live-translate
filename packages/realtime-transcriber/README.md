# realtime-transcriber

macOS のシステム音声をリアルタイムで文字起こし・翻訳する CLI ツールです。

YouTube Live の自動字幕は英語音声・通常レイテンシーの配信のみ対応で、リアルタイムの日本語翻訳は公式機能では実用的ではありません。

Zoom の翻訳字幕は有料プラン限定でホスト依存、Teams・Meet も自社アプリ内でしか動きません。

このツールはシステム音声を直接キャプチャするため、YouTube Live・ウェビナー・Zoom・Teams など任意のアプリの英語音声を、プラットフォームや配信者の設定に関係なくリアルタイムで日本語に翻訳できます。

BlackHole 2ch 経由でキャプチャした音声を MLX-Whisper（`mlx-community/whisper-large-v3-turbo-q4`）で文字起こしし、Amazon Bedrock（デフォルト: Amazon Nova Pro）で日本語に翻訳して表示します。60秒ごとに Amazon Bedrock（Claude Haiku 4.5）で内容を要約し、Whisper の認識精度向上にも活用します。

## アーキテクチャ

```
システム音声 → BlackHole 2ch → sounddevice
  → VAD（Silero VAD）で発話区間を検出
  → MLX-Whisper で文字起こし（要約から生成した英語ヒントで精度向上）
  → ハルシネーション除去（定型フレーズ・繰り返しパターン検出）
  → Amazon Bedrock（Nova Pro）で英→日翻訳（複数文は並列処理、バックエンド切替可能）
  → ターミナルに表示 + ログファイルに記録

  [60秒ごと]
  → Amazon Bedrock（Claude Haiku 4.5）で日本語要約 + 英語キーワード要約を生成
  → 英語キーワード要約を Whisper の initial_prompt に反映
```

## 必要なディスク容量

- MLX-Whisper モデル（`whisper-large-v3-turbo-q4`）: 約 400 MB（初回起動時に `~/.cache/huggingface/hub/` へ自動ダウンロード）
- Python 依存パッケージ: 数百 MB

## 前提条件

- macOS（Apple Silicon）
- Python 3.11〜3.13
- [uv](https://docs.astral.sh/uv/) （パッケージ管理）
- [BlackHole 2ch](https://existential.audio/blackhole/) （仮想オーディオデバイス）
- macOS の「Audio MIDI 設定」で Multi-Output Device を作成済み
- AWS CLI で SSO ログイン済み（`aws sso login`）
- Amazon Bedrock で Claude Haiku 4.5 のモデルアクセスが有効

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

ログファイルは `logs/` ディレクトリにセッションごとに生成されます。

終了は `Ctrl+C` です。

## 出力例

```
● Recording... 3s
  [00:12] Hope you all had a great time at re:Invent so far!
  [00:12] これまでのre:Inventで素晴らしい時間を過ごしたことを願っています!

  [00:18] Raise your hand if you ever been paged at midnight.
  [00:18] 真夜中にページングされたことがある方は手を挙げてください。

--- 要約 ---
• re:Inventセッション。インシデント対応における「コンテキスト麻痺」の問題を提起
• データは溢れているが文脈がないため、MTTRの改善にはMTTCが重要と主張
---
```

## 処理の仕組み

### 音声キャプチャと VAD（発話区間検出）

BlackHole 2ch 経由でシステム音声をステレオで取得し、モノラルに変換して Silero VAD に渡します。VAD が発話区間を検出すると音声バッファへの蓄積を開始し、以下の条件で区切ります。

- 500ms 以上の無音が続いた時点で発話終了と判定
- 発話が 30 秒に達したら強制的に区切り
- 1 秒未満の発話はノイズとして無視

### 文の結合と分割

Whisper の出力が文末（`.` `!` `?` `;`）で終わっていない場合、音声を次のチャンクと結合して再処理します。これにより文の途中で切られることを防ぎます。ただし 15 秒以上蓄積しても文が完結しない場合は、そのまま翻訳に回します（最大で 30 + 15 = 45 秒分の音声がつながる可能性があります）。

文が完結したら、略語（Mr. Dr. e.g. など）のピリオドで誤分割しないよう保護しつつ、文単位に分割して並列翻訳します。

### Whisper のコンテキスト引き継ぎ

Whisper の `initial_prompt` に以下を渡して認識精度を向上させています。

- 直近の文字起こし結果（最大 200 文字、文単位で切り出し）
- 60 秒ごとの要約から生成された英語キーワード（最大 400 文字）

これにより、セッション固有の専門用語や固有名詞が文字起こしに反映されやすくなります。

### ハルシネーション除去

Whisper が無音や短い音声に対して出力する定型フレーズ（"Thank you." "Bye." など）をフィルタリングします。加えて以下のパターンも検出します。

- 同じ単語の 5 回以上の連続（例: "too too too too too"）
- 同じ文字の 10 回以上の連続（例: "llllllllll"）
- 同じ 2〜6 文字パターンの 5 回以上の繰り返し（例: "結論の結論の結論の..."）
- 英数字がほぼ含まれないテキスト

繰り返しが検出された場合、除去後のテキストが元の 50% 未満ならハルシネーションとして破棄し、50% 以上残る場合は繰り返し部分のみ除去して翻訳に回します。

### 定期要約（60 秒ごと）

バックグラウンドのデーモンスレッドが 60 秒ごとに Amazon Bedrock（Claude Haiku 4.5）を呼び出し、前回の要約と直近の翻訳テキストから更新された要約を生成します。1 回の API 呼び出しで以下の 2 つを同時に生成します。

- 日本語要約（ターミナル表示・ログ記録用）: 英語圏特有の表現や略語を補足した自然な文章
- 英語キーワード要約（Whisper の `initial_prompt` 用）: セッションのトピック、専門用語、話者名などを 400 文字以内で記述

### ステータス表示

ターミナルにはリアルタイムで処理状態が表示されます。

- `● Recording... Ns` — VAD が発話を検出し、音声を蓄積中（N は秒数）
- `⏳ Transcribing...` — Whisper が音声を文字起こし中
- `⏳ Translating...` — 翻訳中
- `... waiting` — 文が未完結のため、次のチャンクを待機中

## 翻訳バックエンドの切り替え

`translator.py` の `TRANSLATION_BACKEND` を変更することで翻訳エンジンを切り替えられます。

```python
# Bedrock（デフォルト）— 文脈を考慮した自然な日本語
TRANSLATION_BACKEND = "bedrock"

# AWS Translate — 高速・低コスト。直訳傾向あり
TRANSLATION_BACKEND = "aws_translate"
```

Bedrock の場合、`BEDROCK_MODEL_ID` でモデルも変更できます。

```python
# Amazon Nova Pro（デフォルト、高品質・高速・低コスト）
BEDROCK_MODEL_ID = "us.amazon.nova-pro-v1:0"

# Claude Haiku 4.5（高品質、Nova Proより高価）
BEDROCK_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Amazon Nova Lite（高速、低コスト、品質はやや劣る）
BEDROCK_MODEL_ID = "us.amazon.nova-lite-v1:0"

# Amazon Nova Micro（最速、最安、短文向き）
BEDROCK_MODEL_ID = "us.amazon.nova-micro-v1:0"
```

### 翻訳品質の比較

同じ英語セッション（Apple の re:Invent 登壇）に対して実際に出力された翻訳の比較です。

| 英語原文 | AWS Translate | Nova Lite | Nova Pro（デフォルト） | Haiku 4.5 |
|---------|--------------|-----------|---------------------|-----------|
| power everything from Apple Silicon | **電力を供給** | **動かして** | **動かすために使用** | **動かして** |
| we're a Java shop | **Javaショップ** | **Javaショップ** | **Javaを主に使用** | **Javaを使う企業** |
| can raise eyebrows | 眉をひそめる | 眉をひそめさせる（誤解釈あり） | **注目を集める可能性** | 眉をひそめられる |
| a win for the planet | **地球**にとってもメリット | **地球**にとってより小さい… | お客様と**地球**にとってwin-win | **プラネット**にとっても… |
| homomorphic encryption | ― | 同型暗号化 | **ホモモルフィック暗号化**（補足付き） | 準同型暗号化 |

Bedrock はプロンプトで「日本語話者にわかりやすく、技術用語は英語のまま」と指示しているため、AWS, API, Swift, Graviton などは英語のまま出力します。

### モデル選定の指針

1時間あたりの翻訳コストは、~300回の呼び出し（1回あたり入力 ~150 トークン・出力 ~100 トークン）を想定した目安です。

| モデル | 翻訳品質 | 速度 | 入力 / 100万トークン | 出力 / 100万トークン | 翻訳コスト / 時間 |
|--------|---------|------|---------------------|---------------------|------------------|
| Nova Pro（デフォルト） | Haiku 4.5 と同等 | 速い | $0.80 | $3.20 | ~$0.13 |
| Haiku 4.5 | 高品質 | 普通 | $1.00 | $5.00 | ~$0.20 |
| Nova Lite | やや劣る（慣用表現の誤訳あり） | かなり速い | $0.06 | $0.24 | ~$0.01 |
| Nova Micro | 短文なら実用的 | 最速 | $0.035 | $0.14 | ~$0.005 |
| AWS Translate | 直訳傾向（多義語に弱い） | 最速 | $15.00 / 100万文字 | — | ~$0.08〜$0.23 |

Nova Pro は品質・速度・コストのバランスが最も良いため、デフォルトとしています。

## 主要モジュール

| ファイル | 役割 |
|---------|------|
| `main.py` | CLI エントリポイント。パイプライン全体の制御 |
| `audio.py` | 音声キャプチャと VAD による発話区間検出 |
| `transcriber.py` | MLX-Whisper による文字起こしとハルシネーション除去 |
| `translator.py` | Bedrock または AWS Translate による翻訳（切り替え可能） |
| `summarizer.py` | Amazon Bedrock による定期要約（日本語要約 + Whisper用英語ヒント） |
| `session_logger.py` | セッションごとのログファイル管理 |

## 料金の目安

文字起こし（MLX-Whisper）と VAD（Silero VAD）はローカル実行のため無料です。
AWS の有料サービスは翻訳バックエンドの選択によって異なります。

### Bedrock（Amazon Nova Pro）翻訳 — デフォルト

- 入力: $0.80 / 100万トークン、出力: $3.20 / 100万トークン
- 翻訳: 1回あたり入力 ~150 トークン・出力 ~100 トークン。1時間（~300回）で約 $0.13
- 要約（Haiku 4.5）: 60秒ごと、1回あたり入力 ~600 トークン・出力 ~200 トークン。1時間（60回）で約 $0.07
- 合計目安: 約 **$0.20 / 時間**

### Bedrock（Claude Haiku 4.5）翻訳

- 入力: $1.00 / 100万トークン、出力: $5.00 / 100万トークン
- 翻訳 + 要約で合計目安: 約 **$0.30 / 時間**

### AWS Translate 翻訳

- $15.00 / 100万文字
- 1時間のセッションで約 5,000〜15,000 文字を翻訳すると仮定すると約 $0.08〜$0.23
- Bedrock 要約（$0.07）と合わせて合計目安: 約 **$0.15〜$0.30 / 時間**

## テスト

```bash
uv run pytest
```
