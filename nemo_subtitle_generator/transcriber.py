import os
import tempfile
from enum import Enum
from pathlib import Path

import torch
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
from pydub.silence import detect_silence

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


MAX_SUBTITLE_DURATION = 7.0
MAX_SUBTITLE_WORDS = 20


def _split_long_segment(seg_text: str, words: list[dict], seg_start: float, seg_end: float) -> list[dict]:
    """Split a segment that is too long using word timestamps, preferring punctuation breaks."""
    seg_words = [w for w in words if w['start'] >= seg_start - 0.05 and w['end'] <= seg_end + 0.05]
    if not seg_words:
        return [{"start": seg_start, "end": seg_end, "text": seg_text}]

    subtitles = []
    current_words = []
    current_start = seg_words[0]['start']

    for i, w in enumerate(seg_words):
        if not current_words:
            current_start = w['start']
        current_words.append(w)

        is_last = i == len(seg_words) - 1
        duration = w['end'] - current_start
        word_text = w['word'].rstrip()
        at_comma = word_text.endswith(',')
        too_long = duration > MAX_SUBTITLE_DURATION
        too_many = len(current_words) >= MAX_SUBTITLE_WORDS

        if is_last or too_long or too_many or (at_comma and (duration > 3.0 or len(current_words) >= 6)):
            subtitles.append({
                "start": current_start,
                "end": w['end'],
                "text": " ".join(cw['word'] for cw in current_words),
            })
            current_words = []

    return subtitles


def _extract_segments(hyp) -> list[dict]:
    if not hasattr(hyp, 'timestamp') or not hyp.timestamp:
        return []

    words = hyp.timestamp.get('word', [])
    model_segments = hyp.timestamp.get('segment', [])

    if model_segments and words:
        subtitles = []
        for seg in model_segments:
            text = seg.get('segment', '').strip()
            if not text:
                continue
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            duration = end - start
            word_count = len(text.split())

            if duration <= MAX_SUBTITLE_DURATION and word_count <= MAX_SUBTITLE_WORDS:
                subtitles.append({"start": start, "end": end, "text": text})
            else:
                subtitles.extend(_split_long_segment(text, words, start, end))
        return subtitles

    # Fallback: segment-level only (no word timestamps available)
    if model_segments:
        segments = []
        for seg in model_segments:
            text = seg.get('segment', '').strip()
            if text:
                segments.append({
                    "start": seg.get('start', 0),
                    "end": seg.get('end', 0),
                    "text": text,
                })
        return segments

    # Last resort: word-level with punctuation awareness
    if words:
        subtitles = []
        current_words = []
        current_start = 0.0
        for i, w in enumerate(words):
            if not current_words:
                current_start = w['start']
            current_words.append(w)

            is_last = i == len(words) - 1
            duration = w['end'] - current_start
            word_text = w['word'].rstrip()
            ends_sentence = word_text.endswith(('.', '?', '!'))
            at_comma = word_text.endswith(',')
            has_pause = (i + 1 < len(words) and words[i + 1]['start'] - w['end'] > 0.5)

            should_split = (
                is_last
                or ends_sentence
                or duration > MAX_SUBTITLE_DURATION
                or len(current_words) >= MAX_SUBTITLE_WORDS
                or has_pause
                or (at_comma and (duration > 3.0 or len(current_words) >= 6))
            )

            if should_split:
                subtitles.append({
                    "start": current_start,
                    "end": w['end'],
                    "text": " ".join(cw['word'] for cw in current_words),
                })
                current_words = []
        return subtitles

    return []


def build_srt(hypotheses: list) -> str:
    segments = []
    for hyp in hypotheses:
        segments.extend(_extract_segments(hyp))

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
            self.model.change_subsampling_conv_chunking_factor(1)

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

    def _split_audio(
        self,
        audio: AudioSegment,
        chunk_max_minutes: int,
        chunk_silence_window_s: int,
    ) -> list[tuple[AudioSegment, float]]:
        max_ms = chunk_max_minutes * 60 * 1000
        if len(audio) <= max_ms:
            return [(audio, 0.0)]

        window_ms = chunk_silence_window_s * 1000
        chunks: list[tuple[AudioSegment, float]] = []
        pos = 0

        while pos < len(audio):
            end = pos + max_ms
            if end >= len(audio):
                chunks.append((audio[pos:], pos / 1000.0))
                break

            search_start = max(pos, end - window_ms)
            search_end = min(len(audio), end + window_ms)
            search_region = audio[search_start:search_end]

            silences = detect_silence(search_region, min_silence_len=300, silence_thresh=-40)

            if silences:
                target_in_region = end - search_start
                best_silence = min(
                    silences,
                    key=lambda s: abs((s[0] + s[1]) / 2 - target_in_region),
                )
                cut_in_region = (best_silence[0] + best_silence[1]) // 2
                cut = search_start + cut_in_region
            else:
                cut = end

            chunks.append((audio[pos:cut], pos / 1000.0))
            pos = cut

        return chunks

    def _offset_timestamps(self, hyp, offset_s: float):
        if offset_s == 0.0 or not hasattr(hyp, 'timestamp') or not hyp.timestamp:
            return
        for level in ('word', 'segment', 'char'):
            entries = hyp.timestamp.get(level, [])
            for entry in entries:
                if 'start' in entry:
                    entry['start'] += offset_s
                if 'end' in entry:
                    entry['end'] += offset_s

    def _transcribe_chunks(
        self,
        audio_path: Path,
        timestamps: bool,
        chunking: bool,
        chunk_max_minutes: int,
        chunk_silence_window_s: int,
        **model_kwargs,
    ):
        if chunking and chunk_max_minutes <= 0:
            raise ValueError("chunk_max_minutes must be > 0 when chunking is enabled.")

        audio = AudioSegment.from_file(str(audio_path))
        audio = audio.set_frame_rate(16000).set_channels(1)

        if not chunking:
            split = [(audio, 0.0)]
        else:
            split = self._split_audio(audio, chunk_max_minutes, chunk_silence_window_s)

        all_hypotheses = []
        for chunk_audio, offset_s in split:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                chunk_audio.export(tmp_path, format="wav")
                result = self.model.transcribe(
                    [tmp_path],
                    timestamps=timestamps,
                    **model_kwargs,
                )
                hyp = result[0] if isinstance(result, list) else result
                self._offset_timestamps(hyp, offset_s)
                all_hypotheses.append(hyp)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        return all_hypotheses

    def transcribe(
        self,
        file_path: str | Path,
        source_lang: str = "auto",
        target_lang: str | None = None,
        chunking: bool = True,
        chunk_max_minutes: int = 20,
        chunk_silence_window_s: int = 5,
    ) -> str:
        model_kwargs: dict = {}
        if self.model_type == ModelType.CANARY:
            src, target = self._resolve_canary_langs(source_lang, target_lang)
            model_kwargs.update(source_lang=src, target_lang=target)

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.load_model()
        audio_path, temp_audio = self._prepare_audio(file_path)

        try:
            hypotheses = self._transcribe_chunks(
                audio_path,
                timestamps=False,
                chunking=chunking,
                chunk_max_minutes=chunk_max_minutes,
                chunk_silence_window_s=chunk_silence_window_s,
                **model_kwargs,
            )
            texts = []
            for hyp in hypotheses:
                text = hyp.text if hasattr(hyp, 'text') else str(hyp)
                if text:
                    texts.append(text)
            return " ".join(texts)
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
        chunking: bool = True,
        chunk_max_minutes: int = 20,
        chunk_silence_window_s: int = 5,
    ) -> Path:
        model_kwargs: dict = {}
        if self.model_type == ModelType.CANARY:
            src, target = self._resolve_canary_langs(source_lang, target_lang)
            model_kwargs.update(source_lang=src, target_lang=target)

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
            hypotheses = self._transcribe_chunks(
                audio_path,
                timestamps=True,
                chunking=chunking,
                chunk_max_minutes=chunk_max_minutes,
                chunk_silence_window_s=chunk_silence_window_s,
                **model_kwargs,
            )

            srt_content = build_srt(hypotheses)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            return output_path
        except torch.cuda.OutOfMemoryError:
            self.unload_model()
            raise
        finally:
            if temp_audio and temp_audio.exists():
                os.unlink(temp_audio)
