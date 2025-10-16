# Whisperインストール
pip install openai-whisper

# 音声処理用ライブラリ
pip install ffmpeg-python

# 最小構成

import whisper

# モデルの読み込み
model = whisper.load_model("base")

# 音声ファイルから文字起こし
result = model.transcribe("audio.mp3")

# 結果の出力
print(result["text"])

# 詳細について

import whisper

def transcribe_audio(audio_file_path, model_size="base", language="ja"):
    """
    音声ファイルを文字起こしする関数
    
    Parameters:
    - audio_file_path: 音声ファイルのパス
    - model_size: モデルサイズ ("tiny", "base", "small", "medium", "large")
    - language: 言語コード（"ja"は日本語、"en"は英語など）
    
    Returns:
    - 文字起こし結果の辞書
    """
    
    # モデルの読み込み
    print(f"モデル '{model_size}' を読み込んでいます...")
    model = whisper.load_model(model_size)
    
    # 文字起こし実行
    print(f"ファイル '{audio_file_path}' を文字起こし中...")
    result = model.transcribe(
        audio_file_path,
        language=language,  # 言語を指定
        verbose=True  # 進捗状況を表示
    )
    
    return result


# 実行例
if __name__ == "__main__":
    # 音声ファイルのパス
    audio_path = "sample_audio.mp3"
    
    # 文字起こし実行
    result = transcribe_audio(audio_path, model_size="base", language="ja")
    
    # 結果の表示
    print("\n=== 文字起こし結果 ===")
    print(result["text"])
    
    # テキストファイルに保存
    with open("transcription.txt", "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    print("\n結果を 'transcription.txt' に保存しました。")