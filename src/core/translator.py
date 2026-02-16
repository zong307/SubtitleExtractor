"""Subtitle translation using TranslateGemma model."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

from loguru import logger
try:
    from transformers import pipeline
    import torch
except ImportError:
    logger.error("transformers library not installed. Please install it using: pip install transformers")
    raise


# Language code mapping for TranslateGemma
# See: https://huggingface.co/google/translategemma-4b-it
LANG_CODE_MAP = {
    'en': 'en',
    'zh': 'zh',
    'ja': 'ja',
    'ko': 'ko',
    'fr': 'fr',
    'de': 'de',
    'es': 'es',
    'ru': 'ru',
    'ar': 'ar',
}


class SubtitleTranslator:
    """Translate subtitles using TranslateGemma model."""

    def __init__(self, model_size: str = "4b"):
        """
        Initialize the translator.
        
        Args:
            model_size: Size of the translategemma model ('4b' or '12b')
        """
        self.model_size = model_size
        self.model_name = f"google/translategemma-{model_size}-it"
        self.pipe = None

    def load_model(self) -> None:
        """Load the translation model."""
        logger.info(f"Loading TranslateGemma-{self.model_size} model...")
        
        try:
            # TranslateGemma uses "image-text-to-text" pipeline
            self.pipe = pipeline(
                "image-text-to-text",
                model=self.model_name,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            logger.info(f"TranslateGemma-{self.model_size} model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load TranslateGemma model: {e}")
            raise

    def translate_subtitle_file(self, input_csv_path: str, output_srt_path: str, target_language: str, source_language: str = "auto") -> None:
        """
        Translate subtitle content from CSV to SRT format in the target language.
        
        Args:
            input_csv_path: Path to input CSV file containing subtitles
            output_srt_path: Path to output SRT file
            target_language: Target language code (e.g., 'en', 'zh', 'ja', etc.)
            source_language: Source language code (default: 'auto' for auto-detection)
        """
        if not self.pipe:
            raise RuntimeError("Translation model not loaded. Call load_model() first.")
        
        logger.info(f"Starting translation from {source_language} to {target_language}...")
        
        # Read subtitles from CSV
        subtitles = []
        with open(input_csv_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                subtitles.append({
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'speaker': row.get('speaker', ''),
                    'text': row['text']
                })
        
        if not subtitles:
            logger.warning("No subtitles found in CSV file. Creating empty SRT file.")
            with open(output_srt_path, 'w', encoding='utf-8') as f:
                f.write("")
            return
        
        logger.info(f"Found {len(subtitles)} subtitles to translate.")
        
        # Get target and source language codes
        target_lang_code = LANG_CODE_MAP.get(target_language, target_language)
        source_lang_code = LANG_CODE_MAP.get(source_language, source_language)
        
        # Prepare translated subtitles
        translated_subtitles = []
        for i, subtitle in enumerate(subtitles):
            source_text = subtitle['text']
            if subtitle['speaker']:
                source_text = f"[{subtitle['speaker']}] {source_text}"
            
            # Construct messages following TranslateGemma format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": source_lang_code,
                            "target_lang_code": target_lang_code,
                            "text": source_text,
                        }
                    ],
                }
            ]
            
            # Generate translation using the pipeline
            outputs = self.pipe(text=messages, max_new_tokens=200)
            
            # Extract translated text from output with error handling
            try:
                # Expected format: outputs[0]["generated_text"][-1]["content"]
                translated_text = outputs[0]["generated_text"][-1]["content"]
            except (KeyError, IndexError, TypeError) as e:
                # Debug: log the actual output structure
                logger.warning(f"Unexpected output format: {type(outputs)}, trying alternative parsing...")
                logger.debug(f"Raw output: {outputs}")
                
                # Try alternative parsing methods
                if isinstance(outputs, list) and len(outputs) > 0:
                    first_output = outputs[0]
                    if isinstance(first_output, dict):
                        if "generated_text" in first_output:
                            gen_text = first_output["generated_text"]
                            if isinstance(gen_text, str):
                                translated_text = gen_text
                            elif isinstance(gen_text, list) and len(gen_text) > 0:
                                last_item = gen_text[-1]
                                if isinstance(last_item, dict) and "content" in last_item:
                                    translated_text = last_item["content"]
                                elif isinstance(last_item, str):
                                    translated_text = last_item
                                else:
                                    translated_text = str(last_item)
                            else:
                                translated_text = str(gen_text)
                        else:
                            translated_text = str(first_output)
                    else:
                        translated_text = str(first_output)
                else:
                    translated_text = str(outputs)
            
            # Clean up the translation
            translated_text = self._clean_translation(translated_text)
            
            # Log the translation result immediately
            logger.info(f"[{i+1}/{len(subtitles)}] {source_text} -> {translated_text}")
            
            translated_subtitles.append({
                'index': i + 1,
                'start_time': subtitle['start_time'],
                'end_time': subtitle['end_time'],
                'text': translated_text
            })
        
        # Write translated subtitles to SRT file
        self._write_srt_file(translated_subtitles, output_srt_path)
        logger.info(f"Translation completed. Output saved to: {output_srt_path}")

    def _clean_translation(self, text: str) -> str:
        """Clean up the translated text."""
        # Remove any leading/trailing whitespace and common artifacts
        cleaned = text.strip()
        
        # Remove common prefixes that might appear in the output
        prefixes_to_remove = ["translated:", "translation:", "result:", "output:", ": "]
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
                
        return cleaned

    def _write_srt_file(self, subtitles: list[dict], output_path: str) -> None:
        """
        Write subtitles to SRT format.
        
        Args:
            subtitles: List of subtitle dictionaries
            output_path: Path to output SRT file
        """
        def format_timestamp(seconds_str: str) -> str:
            """Convert seconds to SRT timestamp format."""
            seconds = float(seconds_str)
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        with open(output_path, 'w', encoding='utf-8') as f:
            for i, sub in enumerate(subtitles):
                start_ts = format_timestamp(sub['start_time'])
                end_ts = format_timestamp(sub['end_time'])
                
                f.write(f"{sub['index']}\n")
                f.write(f"{start_ts} --> {end_ts}\n")
                f.write(f"{sub['text']}\n")
                f.write("\n")