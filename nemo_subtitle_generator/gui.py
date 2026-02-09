import os
import tempfile
from pathlib import Path

import gradio as gr

from nemo_subtitle_generator.transcriber import ModelType, Transcriber

transcriber_cache: dict[str, Transcriber] = {}
temp_srt_files: set[Path] = set()


def get_transcriber(model_name: str, local_attention: bool = False) -> Transcriber:
    model_type = ModelType.PARAKEET if "parakeet" in model_name.lower() else ModelType.CANARY
    key = f"{model_type.value}:local_attn={local_attention}"
    if key not in transcriber_cache:
        # Unload other cached models to free VRAM
        for old_key, old_transcriber in list(transcriber_cache.items()):
            old_transcriber.unload_model()
        transcriber_cache.clear()
        transcriber_cache[key] = Transcriber(model_type=model_type, local_attention=local_attention)
    return transcriber_cache[key]


def cleanup_temp_srt_files() -> None:
    for path in list(temp_srt_files):
        try:
            if path.exists():
                os.unlink(path)
        except OSError:
            pass
    temp_srt_files.clear()


def transcribe_file(
    file,
    model_name: str,
    source_lang: str,
    target_lang: str | None,
    output_format: str,
    local_attention: bool,
):
    if file is None:
        return "Please select a file.", None

    transcriber = get_transcriber(model_name, local_attention=local_attention)
    file_path = Path(file.name if hasattr(file, 'name') else file)

    target = target_lang if target_lang and target_lang != "None" else None

    try:
        if output_format == "Plain text":
            text = transcriber.transcribe(
                file_path,
                source_lang=source_lang,
                target_lang=target,
            )
            return text, None

        cleanup_temp_srt_files()
        fd, tmp_path = tempfile.mkstemp(suffix=".srt")
        os.close(fd)
        temp_path = Path(tmp_path)
        temp_srt_files.add(temp_path)
        try:
            srt_path = transcriber.transcribe_to_srt(
                file_path,
                output_path=temp_path,
                source_lang=source_lang,
                target_lang=target,
            )
            with open(srt_path, "r", encoding="utf-8") as f:
                srt_content = f.read()
            return srt_content, str(srt_path)
        except Exception:
            temp_srt_files.discard(temp_path)
            if temp_path.exists():
                os.unlink(temp_path)
            raise
    except Exception as e:
        return f"Error: {e}", None


def transcribe_batch(
    directory: str,
    model_name: str,
    source_lang: str,
    target_lang: str | None,
    max_depth: int,
    local_attention: bool,
    progress=gr.Progress(),
):
    if not directory:
        return "Please enter a directory path."

    dir_path = Path(directory)
    if not dir_path.exists():
        return f"Error: directory not found: {directory}"
    if not dir_path.is_dir():
        return f"Error: {directory} is not a directory"

    transcriber = get_transcriber(model_name, local_attention=local_attention)
    target = target_lang if target_lang and target_lang != "None" else None
    try:
        transcriber.validate_options(source_lang=source_lang, target_lang=target)
    except ValueError as e:
        return f"Error: {e}"

    def find_media_files(path: Path, current_depth: int) -> list[Path]:
        files = []
        if current_depth > max_depth:
            return files
        try:
            for item in path.iterdir():
                if item.is_file() and transcriber.is_supported(item):
                    files.append(item)
                elif item.is_dir():
                    files.extend(find_media_files(item, current_depth + 1))
        except PermissionError:
            pass
        return files

    media_files = find_media_files(dir_path, 0)

    if not media_files:
        return "No audio/video files found."

    results = []
    results.append(f"{len(media_files)} file(s) found\n")

    progress(0, desc="Loading model...")
    transcriber.load_model()

    success_count = 0
    skip_count = 0
    error_count = 0

    for i, file in enumerate(media_files):
        progress((i + 1) / len(media_files), desc=f"Processing {file.name}...")
        output_path = file.with_suffix(".srt")

        if output_path.exists():
            results.append(f"[skip] {file.name} (SRT already exists)")
            skip_count += 1
            continue

        try:
            transcriber.transcribe_to_srt(
                file,
                output_path=output_path,
                source_lang=source_lang,
                target_lang=target,
            )
            results.append(f"[ok] {file.name} -> {output_path.name}")
            success_count += 1
        except Exception as e:
            results.append(f"[error] {file.name}: {e}")
            error_count += 1

    results.append(f"\nDone: {success_count} succeeded")
    if skip_count:
        results.append(f"{skip_count} skipped")
    if error_count:
        results.append(f"{error_count} failed")

    return "\n".join(results)


def create_interface() -> gr.Blocks:
    with gr.Blocks(
        title="Gen Subtitle",
        theme=gr.themes.Soft(),
    ) as interface:
        gr.Markdown("# Gen Subtitle\nSubtitle generation with NVIDIA NeMo")

        with gr.Tabs():
            with gr.TabItem("Single file"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="Audio/video file",
                            file_types=["audio", "video"],
                        )

                        model_dropdown = gr.Dropdown(
                            choices=[
                                "Parakeet TDT 0.6B v3 (25 langs, fast)",
                                "Canary 1B v2 (25 langs, translation)",
                            ],
                            value="Parakeet TDT 0.6B v3 (25 langs, fast)",
                            label="Model",
                        )

                        with gr.Row():
                            source_lang = gr.Dropdown(
                                choices=[
                                    "auto", "bg", "cs", "da", "de", "el", "en", "es", "et",
                                    "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt", "nl",
                                    "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
                                ],
                                value="auto",
                                label="Source language (Parakeet: auto, Canary: explicit)",
                            )
                            target_lang = gr.Dropdown(
                                choices=[
                                    "None", "bg", "cs", "da", "de", "el", "en", "es", "et",
                                    "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt", "nl",
                                    "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
                                ],
                                value="None",
                                label="Translate to (Canary en<->x only)",
                            )

                        output_format = gr.Radio(
                            choices=["SRT file", "Plain text"],
                            value="SRT file",
                            label="Output format",
                        )

                        local_attn = gr.Checkbox(
                            label="Local attention (for long files, reduces VRAM)",
                            value=False,
                        )

                        transcribe_btn = gr.Button("Transcribe", variant="primary")

                    with gr.Column(scale=1):
                        output_text = gr.Textbox(label="Result", lines=15)
                        download_file = gr.File(label="Download SRT", visible=True)

                transcribe_btn.click(
                    fn=transcribe_file,
                    inputs=[file_input, model_dropdown, source_lang, target_lang, output_format, local_attn],
                    outputs=[output_text, download_file],
                )

            with gr.TabItem("Batch"):
                with gr.Row():
                    with gr.Column(scale=1):
                        dir_input = gr.Textbox(
                            label="Directory path",
                            placeholder="/path/to/directory",
                        )

                        batch_model = gr.Dropdown(
                            choices=[
                                "Parakeet TDT 0.6B v3 (25 langs, fast)",
                                "Canary 1B v2 (25 langs, translation)",
                            ],
                            value="Parakeet TDT 0.6B v3 (25 langs, fast)",
                            label="Model",
                        )

                        with gr.Row():
                            batch_source_lang = gr.Dropdown(
                                choices=[
                                    "auto", "bg", "cs", "da", "de", "el", "en", "es", "et",
                                    "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt", "nl",
                                    "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
                                ],
                                value="auto",
                                label="Source language (Parakeet: auto, Canary: explicit)",
                            )
                            batch_target_lang = gr.Dropdown(
                                choices=[
                                    "None", "bg", "cs", "da", "de", "el", "en", "es", "et",
                                    "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt", "nl",
                                    "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
                                ],
                                value="None",
                                label="Translate to (Canary en<->x only)",
                            )

                        batch_max_depth = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Max recursion depth",
                        )

                        batch_local_attn = gr.Checkbox(
                            label="Local attention (for long files, reduces VRAM)",
                            value=False,
                        )

                        batch_btn = gr.Button("Start batch", variant="primary")

                    with gr.Column(scale=1):
                        batch_output = gr.Textbox(label="Result", lines=20)

                batch_btn.click(
                    fn=transcribe_batch,
                    inputs=[dir_input, batch_model, batch_source_lang, batch_target_lang, batch_max_depth, batch_local_attn],
                    outputs=[batch_output],
                )

        gr.Markdown(
            "---\n"
            "| Model | Languages | Notes |\n"
            "|-------|-----------|-------|\n"
            "| Parakeet TDT 0.6B v3 | 25 EU languages | Auto-detect, fast |\n"
            "| Canary 1B v2 | 25 EU languages | Explicit source lang, translation en<->x |"
        )

    return interface


def launch_gui(share: bool = False) -> None:
    interface = create_interface()
    interface.launch(share=share)


if __name__ == "__main__":
    launch_gui()
