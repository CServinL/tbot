#!/usr/bin/env python3
"""
ChatCmd - Command-line Multimodal AI Chat Assistant
Combines text generation and image generation in a single interactive interface
"""

import re
import torch
import json
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import argparse

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from PIL import Image
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
except ImportError as e:
    print(f"❌ Missing required packages. Install with:")
    print(f"pip install torch transformers diffusers pillow accelerate safetensors")
    exit(1)


class ChatCmd:
    """A command-line multimodal AI chat assistant that can generate both text and images"""

    def __init__(self,
                 text_model: str = "MaziyarPanahi/calme-3.1-instruct-78b",
                 image_model: str = "SG161222/Realistic_Vision_V2.0",
                 device: str = "auto",
                 enable_cpu_offload: bool = False,
                 use_text_pipeline: bool = True):

        self.text_model_name = text_model
        self.image_model_name = image_model
        self.enable_cpu_offload = enable_cpu_offload
        self.use_text_pipeline = use_text_pipeline

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🔧 Using device: {self.device}")

        # Create output directory
        self.output_dir = Path("generated_images")
        self.output_dir.mkdir(exist_ok=True)

        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []

        # Initialize models
        self.text_pipe = None
        self.text_model = None
        self.text_tokenizer = None
        self.image_pipe = None

        self._load_models()

        # Image generation keywords
        self.image_keywords = [
            "generate image", "create image", "make image", "draw", "paint", "show me",
            "generate picture", "create picture", "make picture", "visualize",
            "image of", "picture of", "photo of", "illustration of",
            "can you draw", "can you create", "can you generate", "can you make",
            "i want to see", "show me an image", "create an artwork"
        ]

    def _load_models(self):
        """Load both text and image generation models"""
        print("🤖 Loading AI models...")

        # Load text model
        try:
            print(f"📝 Loading text model: {self.text_model_name}")
            if self.use_text_pipeline:
                self.text_pipe = pipeline(
                    "text-generation",
                    model=self.text_model_name,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    max_length=1024
                )
            else:
                self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
                self.text_model = AutoModelForCausalLM.from_pretrained(
                    self.text_model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            print("✅ Text model loaded successfully")

        except Exception as e:
            print(f"❌ Error loading text model: {e}")
            print("💡 Falling back to simple responses")
            self.text_pipe = None
            self.text_model = None

        # Load image model
        try:
            print(f"🎨 Loading image model: {self.image_model_name}")
            self.image_pipe = StableDiffusionPipeline.from_pretrained(
                self.image_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )

            # Optimize image pipeline
            self.image_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.image_pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++"
            )

            if self.device == "cuda":
                if self.enable_cpu_offload:
                    self.image_pipe.enable_model_cpu_offload()
                else:
                    self.image_pipe = self.image_pipe.to(self.device)

                self.image_pipe.enable_attention_slicing()
                self.image_pipe.enable_vae_slicing()

                try:
                    self.image_pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
            else:
                self.image_pipe = self.image_pipe.to(self.device)

            print("✅ Image model loaded successfully")

        except Exception as e:
            print(f"❌ Error loading image model: {e}")
            print("💡 Image generation will be disabled")
            self.image_pipe = None

    def _is_image_request(self, text: str) -> bool:
        """Check if the user is requesting image generation"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.image_keywords)

    def _extract_image_prompt(self, text: str) -> str:
        """Extract the image description from user text"""
        text_lower = text.lower()

        patterns = [
            r"generate (?:an? )?image of (.+)",
            r"create (?:an? )?image of (.+)",
            r"make (?:an? )?image of (.+)",
            r"draw (?:an? )?image of (.+)",
            r"show me (?:an? )?image of (.+)",
            r"(?:can you )?(?:generate|create|make|draw) (.+)",
            r"i want to see (?:an? )?image of (.+)",
            r"(?:image|picture|photo) of (.+)",
            r"visualize (.+)",
            r"show me (.+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).strip()

        return text.strip()

    def generate_text_response(self, user_input: str, max_length: int = 300) -> str:
        """Generate text response using the language model"""
        if not self.text_pipe and not self.text_model:
            return "I apologize, but I'm currently unable to generate text responses due to model loading issues."

        self.conversation_history.append({"role": "user", "content": user_input})

        try:
            if self.use_text_pipeline and self.text_pipe:
                response = self.text_pipe(
                    self.conversation_history,
                    max_length=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.text_pipe.tokenizer.eos_token_id
                )

                if isinstance(response[0]['generated_text'], list):
                    assistant_response = response[0]['generated_text'][-1]['content']
                else:
                    assistant_response = self._extract_assistant_response(response[0]['generated_text'])

            else:
                conversation_text = self._format_conversation()
                inputs = self.text_tokenizer.encode(conversation_text, return_tensors="pt")

                if self.device == "cuda":
                    inputs = inputs.to(self.device)

                with torch.no_grad():
                    outputs = self.text_model.generate(
                        inputs,
                        max_length=len(inputs[0]) + max_length,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.text_tokenizer.eos_token_id
                    )

                assistant_response = self.text_tokenizer.decode(
                    outputs[0][len(inputs[0]):],
                    skip_special_tokens=True
                ).strip()

            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            return assistant_response

        except Exception as e:
            return "I apologize, but I encountered an error while generating a response."

    def generate_image(self, prompt: str, save_image: bool = True) -> Tuple[Optional[Image.Image], str]:
        """Generate image from text prompt"""
        if not self.image_pipe:
            return None, "❌ Image generation is not available (model failed to load)"

        try:
            print(f"🎨 Generating image: '{prompt}'")

            negative_prompt = (
                "blurry, bad anatomy, bad hands, text, error, missing fingers, "
                "extra digit, fewer digits, cropped, worst quality, low quality, "
                "normal quality, jpeg artifacts, signature, watermark, username, "
                "deformed, ugly, mutilated, disfigured, extra limbs"
            )

            with torch.autocast(self.device):
                result = self.image_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=512,
                    height=512,
                    num_inference_steps=25,
                    guidance_scale=7.5
                )

            image = result.images[0]

            if save_image:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_{timestamp}.png"
                filepath = self.output_dir / filename
                image.save(filepath, "PNG")

                metadata = {
                    "prompt": prompt,
                    "timestamp": datetime.now().isoformat(),
                    "model": self.image_model_name,
                    "filename": filename,
                    "width": 512,
                    "height": 512,
                    "steps": 25,
                    "guidance_scale": 7.5
                }

                meta_filepath = self.output_dir / f"generated_{timestamp}.json"
                with open(meta_filepath, 'w') as f:
                    json.dump(metadata, f, indent=2)

                return image, f"✅ Image generated and saved as: {filepath}"
            else:
                return image, "✅ Image generated successfully"

        except Exception as e:
            return None, f"❌ Error generating image: {str(e)}"

    def _format_conversation(self) -> str:
        """Format conversation history for model input"""
        formatted = ""
        for msg in self.conversation_history:
            formatted += f"{msg['role'].capitalize()}: {msg['content']}\n"
        formatted += "Assistant:"
        return formatted

    def _extract_assistant_response(self, generated_text: str) -> str:
        """Extract assistant response from generated text"""
        lines = generated_text.split('\n')
        for line in lines:
            if line.startswith('Assistant:'):
                return line.replace('Assistant:', '').strip()
        return generated_text.strip()

    def process_user_input(self, user_input: str) -> str:
        """Process user input and determine whether to generate text or image"""
        user_input = user_input.strip()

        if self._is_image_request(user_input):
            image_prompt = self._extract_image_prompt(user_input)
            image, message = self.generate_image(image_prompt)

            if image:
                text_response = self.generate_text_response(
                    f"I've generated an image of: {image_prompt}. Please acknowledge this and briefly describe what the user requested."
                )
                return f"{message}\n\n🤖 {text_response}"
            else:
                return message
        else:
            return self.generate_text_response(user_input)

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️ Conversation history cleared")

    def save_conversation(self, filename: str = None):
        """Save conversation to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"

        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "text_model": self.text_model_name,
            "image_model": self.image_model_name,
            "conversation": self.conversation_history
        }

        filepath = Path(filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Conversation saved to {filepath}")


def interactive_chat(assistant: ChatCmd):
    """Run interactive chat session"""
    print("\n" + "=" * 60)
    print("💬✨ CHATCMD - AI CHAT ASSISTANT")
    print("=" * 60)
    print("💬 I can chat with you AND generate images!")
    print("🎨 Just ask me to 'generate an image of...' or 'create a picture of...'")
    print("💡 Commands: 'quit', 'clear', 'save', 'help'")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Thanks for chatting! Goodbye!")
                break
            elif user_input.lower() == 'clear':
                assistant.clear_history()
                continue
            elif user_input.lower().startswith('save'):
                parts = user_input.split(maxsplit=1)
                filename = parts[1] if len(parts) > 1 else None
                assistant.save_conversation(filename)
                continue
            elif user_input.lower() == 'help':
                print("\n📖 Commands:")
                print("  • Type anything to chat")
                print("  • Ask for images: 'generate an image of a sunset'")
                print("  • 'clear' - clear conversation history")
                print("  • 'save [filename]' - save conversation")
                print("  • 'quit' - exit the program")
                print("  • 'help' - show this message")
                continue
            elif not user_input:
                continue

            print("\n🤖 ChatCmd: ", end="", flush=True)
            response = assistant.process_user_input(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n\n🛑 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("💡 Please try again or type 'help' for commands.")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="ChatCmd - Command-line Multimodal AI Chat Assistant")

    parser.add_argument("--text-model", default="MaziyarPanahi/calme-3.1-instruct-78b",
                        help="Text generation model")
    parser.add_argument("--image-model", default="SG161222/Realistic_Vision_V2.0",
                        help="Image generation model")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Enable CPU offloading to save VRAM")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Device to use for inference")
    parser.add_argument("--no-pipeline", action="store_true",
                        help="Use direct model loading instead of pipeline for text")

    args = parser.parse_args()

    print("🚀 Starting ChatCmd...")

    try:
        assistant = ChatCmd(
            text_model=args.text_model,
            image_model=args.image_model,
            device=args.device,
            enable_cpu_offload=args.cpu_offload,
            use_text_pipeline=not args.no_pipeline
        )

        print("✅ ChatCmd ready!")
        interactive_chat(assistant)

    except KeyboardInterrupt:
        print("\n\n👋 Startup interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error initializing ChatCmd: {e}")
        print("💡 Please check your system requirements and internet connection")


if __name__ == "__main__":
    main()