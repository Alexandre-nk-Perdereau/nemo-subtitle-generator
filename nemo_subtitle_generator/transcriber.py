import os
import tempfile
from enum import Enum
from pathlib import Path

import torch
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
CANARY_LANGUAGES = {
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu",
    "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
}


class ModelType(str, Enum):
    PARAKEET = "nvidia/parakeet-tdt-0.6b-v3"
    CANARY = "nvidia/canary-1b-v2"


def format_srt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt(output) -> str:
    hyp = output[0] if isinstance(output, list) and output else output

    segments = []

    # Word-level timestamps: max 7s or 8 words per subtitle
    if hasattr(hyp, 'timestamp') and hyp.timestamp:
        words = hyp.timestamp.get('word', [])
        if words:
            current_segment = []
            current_start = 0.0

            for i, w in enumerate(words):
                if not current_segment:
                    current_start = w['start']
                current_segment.append(w['word'])

                is_last = i == len(words) - 1
                too_long = w['end'] - current_start > 7
                too_many = len(current_segment) >= 8

                if is_last or too_long or too_many:
                    segments.append({
                        "start": current_start,
                        "end": w['end'],
                        "text": " ".join(current_segment),
                    })
                    current_segment = []

    # Fallback: segment-level timestamps
    if not segments and hasattr(hyp, 'timestamp') and hyp.timestamp:
        for seg in hyp.timestamp.get('segment', []):
            text = seg.get('segment', '').strip()
            if text:
                segments.append({
                    "start": seg.get('start', 0),
                    "end": seg.get('end', 0),
                    "text": text,
                })

    if not segments:
        raise ValueError("No usable timestamps found in model output.")

    lines = []
    for i, seg in enumerate(segments, 1):
        if not seg["text"]:
            continue
        start = format_srt_timestamp(seg["start"])
        end = format_srt_timestamp(seg["end"])
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"])
        lines.append("")

    return "\n".join(lines)


class Transcriber:
    def __init__(self, model_type: ModelType = ModelType.PARAKEET, local_attention: bool = False):
        self.model_type = model_type
        self.local_attention = local_attention
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def unload_model(self) -> None:
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self) -> None:
        if self.model is not None:
            return

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_type.value
        )

        if self.local_attention and self.model_type == ModelType.PARAKEET:
            self.model.change_attention_model(
                self_attention_model="rel_pos_local_attn",
                att_context_size=[256, 256],
            )

        self.model = self.model.to(self.device)
        self.model.eval()

    def is_supported(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

    def _extract_audio(self, video_path: Path) -> Path:
        audio = AudioSegment.from_file(str(video_path))
        audio = audio.set_frame_rate(16000).set_channels(1)
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            audio.export(tmp_path, format="wav")
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        return Path(tmp_path)

    def _is_compatible_wav(self, file_path: Path) -> bool:
        if file_path.suffix.lower() != ".wav":
            return False
        try:
            audio = AudioSegment.from_file(str(file_path))
            return audio.frame_rate == 16000 and audio.channels == 1
        except Exception:
            return False

    def _prepare_audio(self, file_path: Path) -> tuple[Path, Path | None]:
        if file_path.suffix.lower() not in VIDEO_EXTENSIONS | AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
        if self._is_compatible_wav(file_path):
            return file_path, None
        audio_path = self._extract_audio(file_path)
        return audio_path, audio_path

    def _resolve_canary_langs(self, source_lang: str, target_lang: str | None) -> tuple[str, str]:
        src = source_lang.strip().lower()
        if src == "auto":
            raise ValueError("Canary requires an explicit source language (e.g. --source-lang fr).")
        if src not in CANARY_LANGUAGES:
            raise ValueError(f"Unsupported Canary source language: {source_lang}")

        target = target_lang.strip().lower() if target_lang else src
        if target not in CANARY_LANGUAGES:
            raise ValueError(f"Unsupported Canary target language: {target_lang}")
        if target != src and src != "en" and target != "en":
            raise ValueError(
                "Canary translation currently supports en<->x pairs only. "
                "Use source==target for transcription."
            )
        return src, target

    def validate_options(self, source_lang: str = "auto", target_lang: str | None = None) -> None:
        if self.model_type == ModelType.CANARY:
            self._resolve_canary_langs(source_lang, target_lang)

    def transcribe(
        self,
        file_path: str | Path,
        source_lang: str = "auto",
        target_lang: str | None = None,
    ) -> str:
        canary_langs: tuple[str, str] | None = None
        if self.model_type == ModelType.CANARY:
            canary_langs = self._resolve_canary_langs(source_lang, target_lang)

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.load_model()
        audio_path, temp_audio = self._prepare_audio(file_path)

        try:
            if self.model_type == ModelType.CANARY:
                src, target = canary_langs if canary_langs is not None else (source_lang, target_lang or source_lang)
                result = self.model.transcribe(
                    [str(audio_path)],
                    source_lang=src,
                    target_lang=target,
                )
            else:
                result = self.model.transcribe([str(audio_path)])

            if isinstance(result, list):
                result = result[0] if result else ""
            if hasattr(result, 'text'):
                result = result.text
            return str(result)
        except torch.cuda.OutOfMemoryError:
            self.unload_model()
            raise
        finally:
            if temp_audio and temp_audio.exists():
                os.unlink(temp_audio)

    def transcribe_to_srt(
        self,
        file_path: str | Path,
        output_path: str | Path | None = None,
        source_lang: str = "auto",
        target_lang: str | None = None,
    ) -> Path:
        canary_langs: tuple[str, str] | None = None
        if self.model_type == ModelType.CANARY:
            canary_langs = self._resolve_canary_langs(source_lang, target_lang)

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if output_path is None:
            output_path = file_path.with_suffix(".srt")
        else:
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.load_model()
        audio_path, temp_audio = self._prepare_audio(file_path)

        try:
            if self.model_type == ModelType.CANARY:
                src, target = canary_langs if canary_langs is not None else (source_lang, target_lang or source_lang)
                output = self.model.transcribe(
                    [str(audio_path)],
                    source_lang=src,
                    target_lang=target,
                    timestamps=True,
                )
            else:
                output = self.model.transcribe(
                    [str(audio_path)],
                    timestamps=True,
                )

            srt_content = build_srt(output)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            return output_path
        except torch.cuda.OutOfMemoryError:
            self.unload_model()
            raise
        finally:
            if temp_audio and temp_audio.exists():
                os.unlink(temp_audio)
