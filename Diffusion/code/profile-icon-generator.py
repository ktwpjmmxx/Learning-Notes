"""
Stable Diffusion Profile Icon Generator
プロフィールアイコン生成用のStable Diffusionアプリケーション

必要なライブラリのインストール（初回のみ）:
pip install torch torchvision diffusers transformers accelerate pillow gradio

Google Colabの場合:
!pip install torch torchvision diffusers transformers accelerate pillow gradio
"""

import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import gradio as gr
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 設定とグローバル変数
# ===============================

# デバイスの設定（GPU利用可能な場合は自動的にGPUを使用）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# モデルの設定
# ファインチューニング用メモ: 
# - 別のモデルを使用する場合はMODEL_IDを変更
# - カスタムモデルの場合は、ローカルパスまたはHugging Faceのリポジトリパスを指定
MODEL_ID = "runwayml/stable-diffusion-v1-5"  # 基本モデル
# MODEL_ID = "CompVis/stable-diffusion-v1-4"  # 代替モデル
# MODEL_ID = "./path/to/your/finetuned/model"  # ファインチューニング済みモデルのパス

# 画像生成パラメータのデフォルト値
DEFAULT_PARAMS = {
    "width": 512,  # プロフィールアイコン用の正方形サイズ
    "height": 512,
    "num_inference_steps": 30,  # ステップ数（品質と速度のバランス）
    "guidance_scale": 7.5,  # プロンプトへの忠実度（7-15が一般的）
    "negative_prompt": "low quality, blurry, distorted, ugly, bad anatomy, bad proportions",
}

# 保存先ディレクトリ
OUTPUT_DIR = "generated_icons"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# モデルの初期化
# ===============================

class IconGenerator:
    def __init__(self, model_id=MODEL_ID):
        """
        アイコン生成クラスの初期化
        
        ファインチューニング用メモ:
        - LoRAやDreamBoothでファインチューニングしたモデルを使用する場合、
          model_idにそのパスを指定
        - 追加のアダプターを使用する場合は、pipe.load_lora_weights()を使用
        """
        self.device = DEVICE
        self.model_id = model_id
        self.pipe = None
        self.load_model()
    
    def load_model(self):
        """モデルのロード"""
        print(f"Loading model: {self.model_id}")
        
        try:
            # パイプラインの初期化
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,  # NSFWチェッカーを無効化（プロフィール画像用）
                requires_safety_checker=False
            )
            
            # スケジューラーの設定（高速化のためDPMSolverを使用）
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # デバイスへの移動
            self.pipe = self.pipe.to(self.device)
            
            # メモリ最適化（GPU使用時）
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                # self.pipe.enable_xformers_memory_efficient_attention()  # xformersインストール時のみ
            
            print("Model loaded successfully!")
            
            # ファインチューニング用メモ:
            # LoRAウェイトをロードする場合:
            # self.pipe.load_lora_weights("path/to/lora/weights")
            # self.pipe.fuse_lora(lora_scale=0.7)
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_icon(self, 
                     prompt, 
                     negative_prompt=DEFAULT_PARAMS["negative_prompt"],
                     num_images=1,
                     steps=DEFAULT_PARAMS["num_inference_steps"],
                     guidance_scale=DEFAULT_PARAMS["guidance_scale"],
                     seed=None,
                     style_preset=None):
        """
        プロフィールアイコンの生成
        
        Args:
            prompt: 生成したい画像の説明
            negative_prompt: 避けたい要素の説明
            num_images: 生成する画像数
            steps: 推論ステップ数
            guidance_scale: プロンプトへの忠実度
            seed: 乱数シード（再現性のため）
            style_preset: スタイルプリセット
        
        Returns:
            生成された画像のリスト
        """
        
        # スタイルプリセットの適用
        if style_preset:
            prompt = self._apply_style_preset(prompt, style_preset)
        
        # プロフィールアイコン用の最適化されたプロンプト
        enhanced_prompt = self._enhance_prompt_for_icon(prompt)
        
        # シード設定（再現性のため）
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 画像生成
        try:
            images = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                height=DEFAULT_PARAMS["height"],
                width=DEFAULT_PARAMS["width"],
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images
            
            # 後処理（プロフィールアイコン用の最適化）
            processed_images = [self._post_process_icon(img) for img in images]
            
            return processed_images
            
        except Exception as e:
            print(f"Error generating image: {e}")
            raise
    
    def _enhance_prompt_for_icon(self, prompt):
        """
        プロフィールアイコン用にプロンプトを最適化
        
        ファインチューニング用メモ:
        - 特定のスタイルに特化したファインチューニングを行う場合、
          ここで追加するキーワードを調整
        """
        icon_keywords = "portrait, centered, symmetrical, clean background, professional, high quality, detailed"
        return f"{prompt}, {icon_keywords}"
    
    def _apply_style_preset(self, prompt, style):
        """スタイルプリセットの適用"""
        style_presets = {
            "anime": "anime style, manga, japanese animation",
            "realistic": "photorealistic, detailed, professional photography",
            "cartoon": "cartoon style, pixar, 3d animation",
            "minimalist": "minimalist, simple, clean lines, flat design",
            "pixel": "pixel art, 8-bit, retro game style",
            "watercolor": "watercolor painting, artistic, soft colors"
        }
        
        if style in style_presets:
            return f"{prompt}, {style_presets[style]}"
        return prompt
    
    def _post_process_icon(self, image):
        """
        生成された画像の後処理
        - リサイズ
        - 円形クロップ（オプション）
        - 品質調整
        """
        # 正方形にクロップ
        width, height = image.size
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        image = image.crop((left, top, right, bottom))
        
        # GitHubプロフィール用のサイズにリサイズ（400x400推奨）
        image = image.resize((400, 400), Image.Resampling.LANCZOS)
        
        return image
    
    def save_image(self, image, prompt):
        """生成された画像を保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{OUTPUT_DIR}/{timestamp}_{safe_prompt}.png"
        image.save(filename, "PNG", quality=95)
        return filename

# ===============================
# Gradioインターフェース
# ===============================

def create_gradio_interface():
    """Gradio UIの作成"""
    
    # ジェネレーターのインスタンス化
    generator = IconGenerator()
    
    def generate_and_save(prompt, style, negative_prompt, num_images, steps, guidance_scale, seed):
        """画像生成と保存を行う関数"""
        
        # シード値の処理（-1の場合はランダム）
        if seed == -1:
            seed = None
        
        # 画像生成
        images = generator.generate_icon(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            style_preset=style
        )
        
        # 画像の保存
        saved_paths = []
        for img in images:
            path = generator.save_image(img, prompt)
            saved_paths.append(path)
            print(f"Saved: {path}")
        
        return images
    
    # UIの構築
    with gr.Blocks(title="Profile Icon Generator") as interface:
        gr.Markdown("""
        # Stable Diffusion Profile Icon Generator
        GitHubなどのプロフィール画像として使用できるアイコンを生成します。
        """)
        
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="プロンプト",
                    placeholder="例: a cute robot with blue eyes, smiling",
                    lines=3
                )
                
                style_dropdown = gr.Dropdown(
                    label="スタイルプリセット",
                    choices=["none", "anime", "realistic", "cartoon", "minimalist", "pixel", "watercolor"],
                    value="none"
                )
                
                with gr.Accordion("詳細設定", open=False):
                    negative_prompt_input = gr.Textbox(
                        label="ネガティブプロンプト",
                        value=DEFAULT_PARAMS["negative_prompt"],
                        lines=2
                    )
                    
                    num_images_slider = gr.Slider(
                        label="生成枚数",
                        minimum=1,
                        maximum=4,
                        value=1,
                        step=1
                    )
                    
                    steps_slider = gr.Slider(
                        label="推論ステップ数",
                        minimum=10,
                        maximum=50,
                        value=DEFAULT_PARAMS["num_inference_steps"],
                        step=5
                    )
                    
                    guidance_slider = gr.Slider(
                        label="ガイダンススケール",
                        minimum=1,
                        maximum=20,
                        value=DEFAULT_PARAMS["guidance_scale"],
                        step=0.5
                    )
                    
                    seed_input = gr.Number(
                        label="シード値 (-1でランダム)",
                        value=-1,
                        precision=0
                    )
                
                generate_btn = gr.Button("生成", variant="primary")
            
            with gr.Column():
                output_gallery = gr.Gallery(
                    label="生成されたアイコン",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto"
                )
        
        # サンプルプロンプト
        gr.Examples(
            examples=[
                ["a friendly robot with glowing eyes, tech avatar"],
                ["cute cat wearing glasses, programmer cat"],
                ["abstract geometric patterns, modern design"],
                ["minimalist mountain landscape, sunset colors"],
                ["cyberpunk character, neon lights, futuristic"]
            ],
            inputs=prompt_input
        )
        
        # イベントハンドラー
        generate_btn.click(
            fn=generate_and_save,
            inputs=[
                prompt_input,
                style_dropdown,
                negative_prompt_input,
                num_images_slider,
                steps_slider,
                guidance_slider,
                seed_input
            ],
            outputs=output_gallery
        )
    
    return interface

# ===============================
# ファインチューニング用の追加情報
# ===============================

"""
ファインチューニングガイド:

1. LoRAを使用したファインチューニング:
   - diffusers の train_text_to_image_lora.py を使用
   - 必要なデータセット: 100-1000枚程度の画像とキャプション
   - 学習率: 1e-4 程度
   - ステップ数: 1000-5000

2. DreamBoothを使用したファインチューニング:
   - 特定のキャラクターやスタイルを学習させる場合に有効
   - 必要なデータセット: 5-20枚程度の画像
   - クラス画像も用意することで品質向上

3. Textual Inversionを使用:
   - 新しいコンセプトを既存モデルに追加
   - 少ないデータセット（3-5枚）で可能

4. データセットの準備:
   - プロフィールアイコンに特化したデータセットを収集
   - 正方形にクロップし、512x512にリサイズ
   - 多様なスタイルとテーマを含める

5. 学習環境:
   - GPU（VRAM 16GB以上推奨）
   - Google Colab Pro/Pro+の利用も可能
"""

# ===============================
# メイン実行部分
# ===============================

if __name__ == "__main__":
    # Gradioインターフェースの起動
    interface = create_gradio_interface()
    
    # ローカル環境での起動
    # interface.launch(share=True)  # share=Trueで公開URL生成
    
    # Google Colabでの起動
    interface.launch(share=True, debug=True)
    
    # コマンドラインでの単体使用例
    # generator = IconGenerator()
    # images = generator.generate_icon(
    #     prompt="a cute robot avatar with blue eyes",
    #     num_images=1
    # )
    # generator.save_image(images[0], "robot_avatar")