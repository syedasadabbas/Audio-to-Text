"""
Desktop GUI for the Audio Transcription Pipeline.

File tab:  batch transcription or Play & Transcribe (word-by-word sync with audio)
Live tab:  real-time mic transcription

Thread safety: only the main thread touches Tk widgets.
  Background workers use queue.Queue + .after() polling.
  The sounddevice callback mutates a single-element list — GIL makes this atomic.
"""

import json
import queue
import sys
import threading
import time
import tkinter as tk
import tkinter.filedialog as fd
import tkinter.messagebox as mb
from pathlib import Path

import customtkinter as ctk
import numpy as np
import sounddevice as sd

_ROOT = str(Path(__file__).parent.parent)
_APP_DIR = str(Path(__file__).parent)
for _p in (_ROOT, _APP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from audio_processor import load_and_normalize
from transcriber import Transcriber, TranscriptionResult
from transcriber_classic import ClassicTranscriber
from realtime_transcriber import RealtimeTranscriber

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

SAMPLE_RATE = 16_000

SUPPORTED_EXTENSIONS = (
    ("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg *.webm *.aac *.opus"),
    ("All files", "*.*"),
)
WHISPER_MODELS = ["tiny", "base", "small", "medium"]


class TranscriptionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Audio Transcription Pipeline")
        self.geometry("880x780")
        self.minsize(720, 620)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._selected_file: str = ""
        self._result: TranscriptionResult | None = None
        self._worker_queue: queue.Queue = queue.Queue()
        self._file_running = False
        # Tracks segments revealed so far; Copy/Export operate on this, not _result.segments.
        # Batch transcribe fills it all at once; Play & Transcribe grows it segment-by-segment.
        self._revealed_segments: list = []

        self._pb_audio: np.ndarray | None = None     # float32 mono samples
        self._pb_pos = [0]                           # [frames_played] — mutated by SD callback
        self._pb_total = 0
        self._pb_stream: sd.OutputStream | None = None
        self._pb_playing = False
        self._pb_events: list[tuple[float, str, bool, bool]] = []
        self._pb_event_idx: int = 0
        self._pb_seg_done_count: int = 0

        self._live_rt: RealtimeTranscriber | None = None
        self._live_running = False
        self._live_segments: list[dict] = []
        self._record_start: float = 0.0

        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.grid(row=0, column=0, padx=24, pady=(16, 4), sticky="ew")
        ctk.CTkLabel(hdr, text="Audio Transcription Pipeline",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(side="left")

        self._tabs = ctk.CTkTabview(self)
        self._tabs.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="nsew")
        self._tabs.add("  File  ")
        self._tabs.add("  Live  ")
        self._build_file_tab(self._tabs.tab("  File  "))
        self._build_live_tab(self._tabs.tab("  Live  "))

    def _build_file_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(2, weight=1)

        ctrl = ctk.CTkFrame(parent)
        ctrl.grid(row=0, column=0, pady=(8, 4), sticky="ew")
        ctrl.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(ctrl, text="Audio File",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=(16, 8), pady=(14, 6), sticky="w")
        frow = ctk.CTkFrame(ctrl, fg_color="transparent")
        frow.grid(row=0, column=1, columnspan=2, padx=(0, 16), pady=(14, 6), sticky="ew")
        frow.grid_columnconfigure(0, weight=1)
        self._file_label = ctk.CTkLabel(frow, text="No file selected",
                                         text_color="gray60", anchor="w")
        self._file_label.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(frow, text="Browse…", width=90,
                      command=self._browse_file).grid(row=0, column=1)

        ctk.CTkLabel(ctrl, text="Backend",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=1, column=0, padx=(16, 8), pady=6, sticky="w")
        self._backend_var = ctk.StringVar(value="whisper")
        brow = ctk.CTkFrame(ctrl, fg_color="transparent")
        brow.grid(row=1, column=1, columnspan=2, padx=(0, 16), pady=6, sticky="w")
        ctk.CTkRadioButton(brow, text="AI  (faster-whisper)",
                           variable=self._backend_var, value="whisper",
                           command=self._on_backend_change).grid(row=0, column=0, padx=(0, 24))
        ctk.CTkRadioButton(brow, text="Classical  (Google Speech API)",
                           variable=self._backend_var, value="classic",
                           command=self._on_backend_change).grid(row=0, column=1)

        self._model_label = ctk.CTkLabel(ctrl, text="Model",
                                          font=ctk.CTkFont(weight="bold"))
        self._model_label.grid(row=2, column=0, padx=(16, 8), pady=6, sticky="w")
        self._model_var = ctk.StringVar(value="base")
        self._model_menu = ctk.CTkOptionMenu(ctrl, variable=self._model_var,
                                              values=WHISPER_MODELS, width=120)
        self._model_menu.grid(row=2, column=1, padx=(0, 8), pady=6, sticky="w")
        ctk.CTkLabel(ctrl,
                     text="tiny=fast  ·  base=balanced  ·  small/medium=best accuracy",
                     text_color="gray60", font=ctk.CTkFont(size=11)).grid(
            row=2, column=2, padx=(0, 16), pady=6, sticky="w")

        ctk.CTkLabel(ctrl, text="Language",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=3, column=0, padx=(16, 8), pady=(6, 14), sticky="w")
        self._lang_entry = ctk.CTkEntry(ctrl, placeholder_text="en  (auto-detect)", width=120)
        self._lang_entry.grid(row=3, column=1, padx=(0, 8), pady=(6, 14), sticky="w")
        ctk.CTkLabel(ctrl, text="Leave blank to auto-detect.  Examples: en · fr · de · es",
                     text_color="gray60", font=ctk.CTkFont(size=11)).grid(
            row=3, column=2, padx=(0, 16), pady=(6, 14), sticky="w")

        arow = ctk.CTkFrame(parent, fg_color="transparent")
        arow.grid(row=1, column=0, pady=(4, 4), sticky="ew")
        arow.grid_columnconfigure(2, weight=1)

        self._transcribe_btn = ctk.CTkButton(
            arow, text="▶  Transcribe",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38, width=155, command=self._start_transcription)
        self._transcribe_btn.grid(row=0, column=0, padx=(0, 8))

        self._play_btn = ctk.CTkButton(
            arow, text="🔊  Play & Transcribe",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38, width=185,
            fg_color="#1a6b3a", hover_color="#145530",
            command=self._start_play_and_transcribe)
        self._play_btn.grid(row=0, column=1, padx=(0, 14))

        self._stop_pb_btn = ctk.CTkButton(
            arow, text="⏹  Stop",
            font=ctk.CTkFont(size=13),
            height=38, width=90,
            fg_color="#7f1f1f", hover_color="#5a1515",
            command=self._stop_playback)
        self._stop_pb_btn.grid(row=0, column=2, sticky="w")
        self._stop_pb_btn.grid_remove()

        self._file_status = ctk.CTkLabel(arow, text="", text_color="gray70", anchor="w")
        self._file_status.grid(row=1, column=0, columnspan=3, pady=(4, 0), sticky="w")

        self._pb_row = ctk.CTkFrame(parent, fg_color="transparent")
        self._pb_row.grid(row=2, column=0, pady=(0, 4), sticky="ew")
        self._pb_row.grid_columnconfigure(1, weight=1)
        self._pb_row.grid_remove()

        self._pb_time_label = ctk.CTkLabel(
            self._pb_row, text="0:00 / 0:00",
            font=ctk.CTkFont(family="Consolas", size=12), width=110)
        self._pb_time_label.grid(row=0, column=0, padx=(4, 8))

        self._pb_bar = ctk.CTkProgressBar(self._pb_row, mode="determinate", height=14)
        self._pb_bar.set(0)
        self._pb_bar.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        self._pb_spinner = ctk.CTkProgressBar(parent, mode="indeterminate")
        self._pb_spinner.grid(row=3, column=0, pady=(0, 4), sticky="ew")
        self._pb_spinner.grid_remove()

        res_frame = ctk.CTkFrame(parent)
        res_frame.grid(row=4, column=0, pady=(0, 0), sticky="nsew")
        parent.grid_rowconfigure(4, weight=1)
        res_frame.grid_columnconfigure(0, weight=1)
        res_frame.grid_rowconfigure(1, weight=1)

        rh = ctk.CTkFrame(res_frame, fg_color="transparent")
        rh.grid(row=0, column=0, padx=12, pady=(10, 4), sticky="ew")
        rh.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(rh, text="Transcript",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, sticky="w")
        btn_row = ctk.CTkFrame(rh, fg_color="transparent")
        btn_row.grid(row=0, column=1)
        self._copy_btn = ctk.CTkButton(btn_row, text="Copy Text", width=95, height=28,
                                        state="disabled", command=self._copy_plain)
        self._copy_btn.grid(row=0, column=0, padx=(0, 8))
        self._export_btn = ctk.CTkButton(btn_row, text="Export JSON", width=95, height=28,
                                          state="disabled", command=self._export_json)
        self._export_btn.grid(row=0, column=1)

        self._results_box = ctk.CTkTextbox(
            res_frame, font=ctk.CTkFont(family="Consolas", size=12),
            wrap="word", state="disabled")
        self._results_box.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self._set_file_placeholder()

    def _browse_file(self):
        path = fd.askopenfilename(title="Select audio file",
                                  filetypes=SUPPORTED_EXTENSIONS)
        if path:
            self._selected_file = path
            self._file_label.configure(text=Path(path).name,
                                        text_color=("gray10", "gray90"))
            self._file_status.configure(text="")

    def _on_backend_change(self):
        is_w = self._backend_var.get() == "whisper"
        self._model_menu.configure(state="normal" if is_w else "disabled")
        self._model_label.configure(
            text_color=("gray10", "gray90") if is_w else "gray50")
        # Play & Transcribe needs word timestamps — not available with Classic backend
        if not self._file_running and not self._pb_playing:
            if is_w:
                self._play_btn.grid()
            else:
                self._play_btn.grid_remove()

    def _start_transcription(self):
        if not self._selected_file:
            mb.showwarning("No file", "Please select an audio file first.")
            return
        if self._file_running:
            return
        self._file_running = True
        self._result = None
        self._revealed_segments = []
        self._sync_action_buttons()
        self._transcribe_btn.configure(state="disabled", text="Running…")
        self._play_btn.configure(state="disabled")
        self._pb_spinner.grid()
        self._pb_spinner.start()
        backend = self._backend_var.get()
        model_str = f"model: {self._model_var.get()}" if backend == "whisper" else "Google Speech API"
        self._file_status.configure(text=f"Transcribing  ·  {model_str}…", text_color="gray70")
        self._clear_results()
        threading.Thread(target=self._file_worker, daemon=True).start()
        self.after(100, self._poll_file_queue)

    def _file_worker(self):
        try:
            lang = self._lang_entry.get().strip() or None
            if self._backend_var.get() == "whisper":
                result = Transcriber(
                    model_size=self._model_var.get()).transcribe(
                    self._selected_file, language=lang)
            else:
                result = ClassicTranscriber(
                    language=lang or "en-US").transcribe(self._selected_file)
            self._worker_queue.put(("ok", result))
        except Exception as exc:
            self._worker_queue.put(("error", str(exc)))

    def _poll_file_queue(self):
        try:
            event, payload = self._worker_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_file_queue)
            return
        self._pb_spinner.stop()
        self._pb_spinner.grid_remove()
        self._transcribe_btn.configure(state="normal", text="▶  Transcribe")
        self._play_btn.configure(state="normal")
        self._file_running = False
        if event == "ok":
            self._result = payload
            self._revealed_segments = list(payload.segments)
            self._sync_action_buttons()
            self._render_all_segments(payload.segments)
            model_str = f"model: {self._model_var.get()}" if self._backend_var.get() == "whisper" \
                else "Google Speech API"
            self._file_status.configure(
                text=f"Done  ·  {len(payload.segments)} segments  ·  "
                     f"{payload.duration:.1f}s  ·  lang: {payload.language}  ·  {model_str}",
                text_color="gray70")
        else:
            self._file_status.configure(text=f"Error: {payload}", text_color="#e74c3c")
            mb.showerror("Transcription failed", payload)

    def _start_play_and_transcribe(self):
        if not self._selected_file:
            mb.showwarning("No file", "Please select an audio file first.")
            return
        if self._file_running or self._pb_playing:
            return

        self._file_running = True
        self._result = None
        self._revealed_segments = []
        self._sync_action_buttons()
        self._transcribe_btn.configure(state="disabled")
        self._play_btn.configure(state="disabled", text="Transcribing…")
        self._pb_spinner.grid()
        self._pb_spinner.start()
        model_str = self._model_var.get()
        self._file_status.configure(
            text=f"Transcribing  ·  model: {model_str}  — playback starts when ready…",
            text_color="gray70")
        self._clear_results()

        threading.Thread(target=self._play_transcribe_worker, daemon=True).start()
        self.after(100, self._poll_play_queue)

    def _play_transcribe_worker(self):
        try:
            lang = self._lang_entry.get().strip() or None
            result = Transcriber(model_size=self._model_var.get()).transcribe(
                self._selected_file, language=lang)

            pydub_audio = load_and_normalize(self._selected_file)
            samples = np.array(pydub_audio.get_array_of_samples(),
                               dtype=np.float32) / 32768.0

            self._worker_queue.put(("play_ready", (result, samples)))
        except Exception as exc:
            self._worker_queue.put(("error", str(exc)))

    def _poll_play_queue(self):
        try:
            event, payload = self._worker_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_play_queue)
            return

        self._pb_spinner.stop()
        self._pb_spinner.grid_remove()
        self._file_running = False

        if event == "play_ready":
            result, samples = payload
            self._result = result
            self._file_status.configure(
                text=f"Playing  ·  {len(result.segments)} segments  ·  "
                     f"{result.duration:.1f}s  ·  lang: {result.language}",
                text_color="#2ecc71")
            self._launch_playback(samples, result.segments)
        else:
            self._transcribe_btn.configure(state="normal")
            self._play_btn.configure(state="normal", text="🔊  Play & Transcribe")
            self._file_status.configure(text=f"Error: {payload}", text_color="#e74c3c")
            mb.showerror("Failed", payload)

    def _build_caption_events(
        self, segments: list
    ) -> list[tuple[float, str, bool, bool]]:
        """
        Flatten segments into per-word events: (timestamp, text, is_seg_start, is_seg_end).
        Falls back to one event per segment if word timestamps aren't available.
        """
        events: list[tuple[float, str, bool, bool]] = []
        for seg in segments:
            if seg.words:
                for i, w in enumerate(seg.words):
                    is_first = (i == 0)
                    is_last = (i == len(seg.words) - 1)
                    events.append((w.start, w.text, is_first, is_last))
            else:
                events.append((seg.start, seg.text.strip(), True, True))
        return events

    def _launch_playback(self, samples: np.ndarray, segments: list):
        self._pb_audio = samples
        self._pb_total = len(samples)
        self._pb_pos = [0]
        self._pb_events = self._build_caption_events(segments)
        self._pb_event_idx = 0
        self._pb_seg_done_count = 0
        self._pb_playing = True
        self._pb_segments_ref = segments

        self._play_btn.configure(state="disabled", text="🔊  Playing…")
        self._stop_pb_btn.grid()
        self._pb_row.grid()
        self._pb_time_label.configure(text=f"0:00 / {self._fmt(self._pb_total / SAMPLE_RATE)}")
        self._pb_bar.set(0)

        def _sd_callback(outdata, frames, _time_info, _status):
            pos = self._pb_pos[0]
            chunk = self._pb_audio[pos: pos + frames]
            if len(chunk) < frames:
                outdata[:len(chunk), 0] = chunk
                outdata[len(chunk):, 0] = 0.0
                self._pb_pos[0] = self._pb_total
                raise sd.CallbackStop()
            outdata[:, 0] = chunk
            self._pb_pos[0] = pos + frames

        self._pb_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=1024, callback=_sd_callback,
            finished_callback=self._on_playback_finished)
        self._pb_stream.start()
        self.after(40, self._playback_tick)

    def _playback_tick(self):
        if not self._pb_playing:
            return

        current_sec = self._pb_pos[0] / SAMPLE_RATE
        total_sec = self._pb_total / SAMPLE_RATE

        self._pb_bar.set(current_sec / total_sec if total_sec > 0 else 0)
        self._pb_time_label.configure(
            text=f"{self._fmt(current_sec)} / {self._fmt(total_sec)}")

        box = self._results_box
        box.configure(state="normal")
        while (self._pb_event_idx < len(self._pb_events) and
               self._pb_events[self._pb_event_idx][0] <= current_sec + 0.05):
            ev = self._pb_events[self._pb_event_idx]
            word, is_seg_start, is_seg_end = ev[1], ev[2], ev[3]

            if is_seg_start:
                seg = self._pb_segments_ref[
                    sum(1 for e in self._pb_events[:self._pb_event_idx] if e[2])
                ]
                conf = getattr(seg, "confidence", None)
                conf_str = f"  {conf:.0%}" if conf is not None else ""
                box.insert("end", f"[{self._fmt(seg.start)} → {self._fmt(seg.end)}]{conf_str}\n  ")

            box.insert("end", word + ("\n\n" if is_seg_end else " "))
            box.see("end")
            if is_seg_end:
                if self._pb_seg_done_count < len(self._pb_segments_ref):
                    self._revealed_segments.append(
                        self._pb_segments_ref[self._pb_seg_done_count])
                    self._sync_action_buttons()
                self._pb_seg_done_count += 1
            self._pb_event_idx += 1

        box.configure(state="disabled")
        self.after(40, self._playback_tick)

    def _on_playback_finished(self):
        self.after(0, self._finalize_playback)

    def _finalize_playback(self):
        self._pb_playing = False
        if self._pb_stream:
            self._pb_stream.close()
            self._pb_stream = None

        # Flush any words not yet revealed when audio ended
        box = self._results_box
        box.configure(state="normal")
        while self._pb_event_idx < len(self._pb_events):
            ev = self._pb_events[self._pb_event_idx]
            word, is_seg_start, is_seg_end = ev[1], ev[2], ev[3]
            if is_seg_start:
                seg = self._pb_segments_ref[
                    sum(1 for e in self._pb_events[:self._pb_event_idx] if e[2])
                ]
                conf = getattr(seg, "confidence", None)
                conf_str = f"  {conf:.0%}" if conf is not None else ""
                box.insert("end", f"[{self._fmt(seg.start)} → {self._fmt(seg.end)}]{conf_str}\n  ")
            box.insert("end", word + ("\n\n" if is_seg_end else " "))
            if is_seg_end:
                if self._pb_seg_done_count < len(self._pb_segments_ref):
                    self._revealed_segments.append(
                        self._pb_segments_ref[self._pb_seg_done_count])
                self._pb_seg_done_count += 1
            self._pb_event_idx += 1
        box.configure(state="disabled")

        self._pb_bar.set(1.0)
        self._play_btn.configure(state="normal", text="🔊  Play & Transcribe")
        self._transcribe_btn.configure(state="normal")
        self._stop_pb_btn.grid_remove()
        self._file_status.configure(text="Playback complete.", text_color="gray70")

    def _stop_playback(self):
        if not self._pb_playing:
            return
        self._pb_playing = False
        if self._pb_stream:
            self._pb_stream.stop()
            self._pb_stream.close()
            self._pb_stream = None
        self._stop_pb_btn.grid_remove()
        self._play_btn.configure(state="normal", text="🔊  Play & Transcribe")
        self._transcribe_btn.configure(state="normal")
        self._file_status.configure(text="Playback stopped.", text_color="gray70")

    def _append_segment_to_box(self, seg):
        ts = f"[{self._fmt(seg.start)} → {self._fmt(seg.end)}]"
        conf = getattr(seg, "confidence", None)
        conf_str = f"  {conf:.0%}" if conf is not None else ""
        self._results_box.configure(state="normal")
        self._results_box.insert("end", f"{ts}{conf_str}\n")
        self._results_box.insert("end", f"  {seg.text.strip()}\n\n")
        self._results_box.see("end")
        self._results_box.configure(state="disabled")

    def _render_all_segments(self, segments):
        self._results_box.configure(state="normal")
        self._results_box.delete("1.0", "end")
        if not segments:
            self._results_box.insert("end", "No speech detected.\n")
        else:
            for seg in segments:
                self._append_segment_to_box(seg)
        self._results_box.configure(state="disabled")

    def _set_file_placeholder(self):
        self._results_box.configure(state="normal")
        self._results_box.insert("end",
            "Results appear here.\n\n"
            "  ▶ Transcribe          — process file, show all results at once\n"
            "  🔊 Play & Transcribe  — play audio, segments appear as they're spoken")
        self._results_box.configure(state="disabled")

    def _clear_results(self):
        self._results_box.configure(state="normal")
        self._results_box.delete("1.0", "end")
        self._results_box.configure(state="disabled")

    def _sync_action_buttons(self):
        state = "normal" if self._revealed_segments else "disabled"
        self._copy_btn.configure(state=state)
        self._export_btn.configure(state=state)

    def _copy_plain(self):
        self.clipboard_clear()
        self.clipboard_append(" ".join(s.text.strip() for s in self._revealed_segments))
        count = len(self._revealed_segments)
        total = len(self._result.segments) if self._result else count
        note = f" ({count}/{total} segments)" if count < total else ""
        self._file_status.configure(text=f"Copied to clipboard{note}.")

    def _export_json(self):
        data = {
            "file": self._result.file_path,
            "language": self._result.language,
            "duration_seconds": round(self._result.duration, 2),
            "segments_revealed": len(self._revealed_segments),
            "segments_total": len(self._result.segments),
            "segments": [
                {
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "text": s.text.strip(),
                    "confidence": round(getattr(s, "confidence", 0), 3),
                }
                for s in self._revealed_segments
            ],
        }
        path = fd.asksaveasfilename(
            initialfile=Path(self._selected_file).stem + "_transcript.json",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            count = len(self._revealed_segments)
            total = len(self._result.segments)
            note = f" ({count}/{total} segments)" if count < total else ""
            self._file_status.configure(text=f"Saved → {Path(path).name}{note}")

    def _build_live_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(2, weight=1)

        cfg = ctk.CTkFrame(parent)
        cfg.grid(row=0, column=0, pady=(8, 4), sticky="ew")
        cfg.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(cfg, text="Model",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=(16, 8), pady=(14, 14), sticky="w")
        self._live_model_var = ctk.StringVar(value="base")
        ctk.CTkOptionMenu(cfg, variable=self._live_model_var,
                          values=WHISPER_MODELS, width=120).grid(
            row=0, column=1, padx=(0, 16), pady=(14, 14), sticky="w")
        ctk.CTkLabel(cfg,
                     text="Use  tiny  for fastest response.  Model loads once when recording starts.",
                     text_color="gray60", font=ctk.CTkFont(size=11)).grid(
            row=0, column=2, padx=(0, 16), pady=(14, 14), sticky="w")

        live_ctrl = ctk.CTkFrame(parent, fg_color="transparent")
        live_ctrl.grid(row=1, column=0, pady=6, sticky="ew")
        live_ctrl.grid_columnconfigure(2, weight=1)

        self._record_btn = ctk.CTkButton(
            live_ctrl, text="⏺  Start Recording",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40, width=185,
            fg_color="#c0392b", hover_color="#a93226",
            command=self._toggle_recording)
        self._record_btn.grid(row=0, column=0, padx=(0, 14))

        self._timer_label = ctk.CTkLabel(
            live_ctrl, text="00:00",
            font=ctk.CTkFont(family="Consolas", size=20, weight="bold"),
            text_color="gray50")
        self._timer_label.grid(row=0, column=1, padx=(0, 14))

        level_col = ctk.CTkFrame(live_ctrl, fg_color="transparent")
        level_col.grid(row=0, column=2, sticky="ew", padx=(0, 12))
        level_col.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(level_col, text="Mic level",
                     font=ctk.CTkFont(size=11), text_color="gray60").grid(
            row=0, column=0, sticky="w")
        self._level_bar = ctk.CTkProgressBar(level_col, height=10)
        self._level_bar.set(0)
        self._level_bar.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        self._live_status = ctk.CTkLabel(
            live_ctrl, text="Ready — press Start to begin",
            text_color="gray60", font=ctk.CTkFont(size=12))
        self._live_status.grid(row=1, column=0, columnspan=3, pady=(6, 0), sticky="w")

        lt_frame = ctk.CTkFrame(parent)
        lt_frame.grid(row=2, column=0, pady=(4, 0), sticky="nsew")
        lt_frame.grid_columnconfigure(0, weight=1)
        lt_frame.grid_rowconfigure(1, weight=1)

        lh = ctk.CTkFrame(lt_frame, fg_color="transparent")
        lh.grid(row=0, column=0, padx=12, pady=(10, 4), sticky="ew")
        lh.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(lh, text="Live Transcript",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, sticky="w")
        btn_live = ctk.CTkFrame(lh, fg_color="transparent")
        btn_live.grid(row=0, column=1)
        ctk.CTkButton(btn_live, text="Copy Text", width=95, height=28,
                      command=self._copy_live).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkButton(btn_live, text="Clear", width=75, height=28,
                      command=self._clear_live).grid(row=0, column=1)

        self._live_box = ctk.CTkTextbox(
            lt_frame, font=ctk.CTkFont(family="Consolas", size=12),
            wrap="word", state="disabled")
        self._live_box.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self._live_box.configure(state="normal")
        self._live_box.insert("end",
            "Live transcription will appear here as you speak.\n\n"
            "Press  ⏺ Start Recording  to begin.")
        self._live_box.configure(state="disabled")

    def _toggle_recording(self):
        if self._live_running:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        self._live_running = True
        self._live_segments.clear()
        self._record_start = time.time()
        self._record_btn.configure(text="⏹  Stop Recording",
                                    fg_color="#27ae60", hover_color="#1e8449")
        self._live_status.configure(text="Listening…  speak now", text_color="#2ecc71")
        self._timer_label.configure(text_color=("#27ae60", "#2ecc71"))
        self._live_box.configure(state="normal")
        self._live_box.delete("1.0", "end")
        self._live_box.insert("end", "Listening…\n")
        self._live_box.configure(state="disabled")
        self._live_rt = RealtimeTranscriber(
            on_segment=self._on_live_segment,
            model_size=self._live_model_var.get())
        self._live_rt.start()
        self._update_timer()
        self._update_level_meter()

    def _stop_recording(self):
        self._live_running = False
        if self._live_rt:
            threading.Thread(target=self._live_rt.stop, daemon=True).start()
            self._live_rt = None
        self._record_btn.configure(text="⏺  Start Recording",
                                    fg_color="#c0392b", hover_color="#a93226")
        self._live_status.configure(
            text=f"Stopped  ·  {len(self._live_segments)} segments captured",
            text_color="gray60")
        self._timer_label.configure(text_color="gray50")
        self._level_bar.set(0)

    def _on_live_segment(self, start, end, text, final):
        self.after(0, self._append_live_segment, start, end, text, final)

    def _append_live_segment(self, start, end, text, final):
        self._live_segments.append({"start": start, "end": end, "text": text})
        self._live_box.configure(state="normal")
        content = self._live_box.get("1.0", "end").strip()
        if content in ("Listening…", ""):
            self._live_box.delete("1.0", "end")
        suffix = "\n\n" if final else "  …\n\n"
        self._live_box.insert("end", f"[{self._fmt(start)} → {self._fmt(end)}]\n")
        self._live_box.insert("end", f"  {text.strip()}{suffix}")
        self._live_box.see("end")
        self._live_box.configure(state="disabled")
        n = len(self._live_segments)
        self._live_status.configure(
            text=f"Listening…  {n} segment{'s' if n != 1 else ''} captured")

    def _update_timer(self):
        if not self._live_running:
            return
        e = int(time.time() - self._record_start)
        self._timer_label.configure(text=f"{e // 60:02d}:{e % 60:02d}")
        self.after(1000, self._update_timer)

    def _update_level_meter(self):
        if not self._live_running:
            return
        level = 0.0
        if self._live_rt is not None:
            level = min(1.0, getattr(self._live_rt, "_last_rms", 0.0) * 20)
        self._level_bar.set(level)
        self.after(80, self._update_level_meter)

    def _copy_live(self):
        if not self._live_segments:
            mb.showinfo("Nothing to copy", "No live transcript yet.")
            return
        self.clipboard_clear()
        self.clipboard_append(" ".join(s["text"].strip() for s in self._live_segments))
        self._live_status.configure(text="Copied to clipboard.")

    def _clear_live(self):
        self._live_segments.clear()
        self._live_box.configure(state="normal")
        self._live_box.delete("1.0", "end")
        self._live_box.insert("end", "Cleared.  Press  ⏺ Start Recording  to begin again.")
        self._live_box.configure(state="disabled")
        self._live_status.configure(text="Ready — press Start to begin", text_color="gray60")

    @staticmethod
    def _fmt(seconds: float) -> str:
        m = int(seconds) // 60
        s = seconds % 60
        return f"{m}:{s:05.2f}" if m else f"{s:.2f}s"

    def _on_close(self):
        if self._live_running:
            self._stop_recording()
        if self._pb_playing:
            self._stop_playback()
        self.destroy()


if __name__ == "__main__":
    app = TranscriptionApp()
    app.mainloop()
