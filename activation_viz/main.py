import itertools
import os
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from logging import getLogger

import customtkinter as ctk

from activation_viz.fixtures import TOKENS, get_default_graph_text
from activation_viz.llm import ProfiledSmolLM, ProfiledToken, download_model

USE_MOCK_LLM = os.environ.get("USE_MOCK_LLM", "").lower() in ("1", "true", "yes")

logger = getLogger(__name__)

# CTkTextbox needs an explicit monospace font so the activation graph's ASCII grid stays aligned.
MONOSPACE_FONT = {"family": "Courier New", "size": 13}

LLM_INPUT_PLACEHOLDER = "What is the capital of France?"
# CTkTextbox has no placeholder_text option (unlike CTkEntry), so the placeholder is emulated
# with a dimmed color that's swapped for the real text color on focus in/out.
PLACEHOLDER_TEXT_COLOR = "gray50"


def _char_size(font: ctk.CTkFont, chars: int, lines: int = 1) -> tuple[int, int]:
    """
    Convert a char/line count (as tk.Text/ttk widths used to be given) into the pixel
    width/height CTk widgets expect, measured against the actual font a widget will render
    with -- CTk applies its own DPI/widget scaling on top of the font, so a fixed px-per-char
    guess drifts from the real rendered size across systems.
    """
    return font.measure("0" * chars), font.metrics("linespace") * lines


@dataclass
class GraphConfig:
    """Represents configurations for LLM activations graph"""

    x_size: int
    y_size: int
    threshold: float  # activation threshold


class TokenManager:
    """Handles GUI state and rendering changes"""

    def __init__(
        self,
        root: ctk.CTk,
        graph_config: GraphConfig,
    ) -> None:
        """Initialize GUI state for the given Tk root and graph configuration."""
        self.root = root
        self.graph_config = graph_config
        self.curr_token: int = 0
        self.last_token: int = 0
        self.is_playing = False
        self.tokens: list[ProfiledToken] = []

        self.tk_graph: ctk.CTkTextbox
        self.tk_token_status: ctk.CTkLabel
        self.tk_cb_activation: ctk.CTkComboBox
        self.tk_llm_output: ctk.CTkTextbox
        self.tk_toggle_play: ctk.CTkButton
        self.tk_forward: ctk.CTkButton
        self.tk_backward: ctk.CTkButton

    def set_tokens(self, tokens: list[ProfiledToken]) -> None:
        """Load a new list of tokens and start playback from the first token."""
        assert len(tokens) > 0
        assert self.tk_cb_activation

        self.curr_token = 0
        self.last_token = len(tokens) - 1
        self.tokens = tokens

        self.tk_cb_activation.configure(state=ctk.NORMAL)

        self.render_token()
        self.toggle_play()

    def next_token(self) -> None:
        """Advance to and render the next token, if one exists."""
        if self.curr_token >= self.last_token:
            return

        self.curr_token += 1

        if self.curr_token == self.last_token:
            self.toggle_play()

        self.render_token()

    def prior_token(self) -> None:
        """Step back to and render the previous token, if one exists."""
        if self.curr_token > 0:
            self.curr_token -= 1

        self.render_token()

    def render_token(self) -> None:
        """Render the current token's activation graph, output text, and status label."""
        if self.tk_graph is None:
            raise ValueError("tk_graph must be set before using render_token")
        if self.tk_llm_output is None:
            raise ValueError("tk_llm_output must be set before using render_token")
        if self.tk_token_status is None:
            raise ValueError("tk_token_status must be set before using render_token")

        # Render tensors on graph
        compressed_tensors = self._compress_tensors()
        graph_text: str = self._graph_weights(compressed_tensors)
        self.tk_graph.configure(state=ctk.NORMAL)
        self.tk_graph.delete("1.0", "end")
        self.tk_graph.insert("1.0", graph_text)
        self.tk_graph.configure(state=ctk.DISABLED)

        # Render llm output box
        llm_text_tokens = [t.text for t in self.tokens[0 : self.curr_token + 1]]
        llm_output = "".join(llm_text_tokens)
        self.tk_llm_output.configure(state=ctk.NORMAL)
        self.tk_llm_output.delete("1.0", "end")
        self.tk_llm_output.insert("1.0", llm_output)
        self.tk_llm_output.configure(state=ctk.DISABLED)

        # Render token x / x label
        self.tk_token_status.configure(text=f"token {self.curr_token} / {self.last_token}")

    def toggle_play(self) -> None:
        """Toggle between playing and paused playback state."""
        if self.is_playing:
            # Switch state to paused
            self.is_playing = False
            self.tk_toggle_play.configure(text="▶ Play")
            self.tk_backward.configure(state="normal")
            self.tk_forward.configure(state="normal")
            return

        self.is_playing = True
        self.tk_toggle_play.configure(text="⏸ Pause")
        self.tk_backward.configure(state="disabled")
        self.tk_forward.configure(state="disabled")

        self.root.after(500, self._playback)

    def _playback(self) -> None:
        """Advance playback one token at a time on a timer, while playing."""
        if self.is_playing is False:
            return

        self.next_token()
        self.root.after(250, self._playback)

    def _compress_tensors(self) -> list[list[float]]:
        """Downsample the current token's tensors to fit the graph's y_size dimension."""
        compressed_tensors = []

        # Compress tensors using the mean per batch of activation weights
        for i, tensor in enumerate(self.tokens[self.curr_token].tensors):
            size = len(tensor)

            if size % self.graph_config.y_size > 0:
                raise ValueError(f"Tensor {i} doesn't evenly divide into graph with x_size {self.graph_config.x_size}")

            # Batch tensor activations so we can display them on graph
            # Example: Tensor with 768 activations and a graph with y_size 13 results in batches of 60
            batch_size = size // self.graph_config.y_size

            batched = itertools.batched(tensor, batch_size, strict=True)
            compressed_tensor = []

            for batch in batched:
                batch_mean = sum(batch) / len(batch)
                compressed_tensor.append(batch_mean)

            compressed_tensors.append(compressed_tensor)

        return compressed_tensors

    def _graph_weights(self, tensors: list[list[float]]) -> str:
        """Given list of tensors matching graph dimensions, returns text of activations exceeding threshold"""
        assert len(tensors) == self.graph_config.x_size
        for tensor in tensors:
            assert len(tensor) == self.graph_config.y_size
        assert self.graph_config.x_size < 100  # 100+ messes up column header formatting

        text = ""

        # Generate x-axis header
        for col in range(self.graph_config.x_size):
            col_name = str(col)
            if col < 10:
                col_name = f"0{col}"
            text = f"{text} {col_name}"

        text = f"{text}\n"

        # Render activations as . for no activation and X for activation
        for row in range(self.graph_config.x_size):
            activations = [tensor[row] for tensor in tensors]
            for activation in activations:
                if activation > self.graph_config.threshold:
                    text = f"{text}  X"
                else:
                    text = f"{text}  ."
            text = f"{text}\n"

        return text


def main() -> None:
    """Build and run the activation-viz Tk application."""
    if not USE_MOCK_LLM:
        logger.info("Fetching model...")
        download_model()
        time.sleep(2)

    root = ctk.CTk()

    main = ctk.CTkFrame(root, fg_color="transparent")
    main.grid(sticky="nsew", padx=10, pady=10)
    main.columnconfigure(0, weight=0)
    main.columnconfigure(1, weight=1)

    graph_config = GraphConfig(30, 32, 0.33)
    mgr = TokenManager(root, graph_config)

    # Build the UI at customtkinter's unscaled default (1.0) first, then scale it as a whole --
    # see _scale_and_center for why this has to happen in that order.
    create_display(main, mgr)
    create_sidebar(main, mgr)

    _scale_and_center(root)

    main.mainloop()


def _scale_and_center(root: ctk.CTk) -> None:
    """
    Scale the already-built UI to the display's real DPI and center the window on screen.

    customtkinter doesn't auto-detect DPI scaling on X11 (unlike classic Tk, which does), so
    without this its widgets render far smaller than the rest of the desktop. The fix is
    ctk.set_widget_scaling(), but that call has a side effect: it immediately locks the real
    window's min/max size to whatever CTk's tracked "current size" is at that moment. CTk
    starts every window at a hardcoded 600x500 regardless of content, and on at least some
    window managers that lock, once set, can't be widened again by any later call -- resizing
    or rescaling -- which is what cropped the window before this fix. So scaling is applied
    only once, here, after layout, with CTk's tracked size seeded to the real unscaled content
    size first, so the one-and-only lock already matches what the window actually needs.
    """
    root.update_idletasks()
    unscaled_w, unscaled_h = root.winfo_reqwidth(), root.winfo_reqheight()
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    margin_w, margin_h = 80, 120  # room for window decorations/taskbars

    dpi_scale = root.winfo_fpixels("1i") / 96  # 96 DPI is customtkinter's "100%" baseline
    scale = min(dpi_scale, (screen_w - margin_w) / unscaled_w, (screen_h - margin_h) / unscaled_h)

    root._current_width, root._current_height = unscaled_w, unscaled_h
    ctk.set_widget_scaling(scale)
    ctk.set_window_scaling(scale)
    root.update_idletasks()

    x = max((screen_w - round(unscaled_w * scale)) // 2, 0)
    y = max((screen_h - round(unscaled_h * scale)) // 2, 0)
    root.geometry(f"+{x}+{y}")  # position only -- see docstring for why not to resize again here


def create_sidebar(frm: ctk.CTkFrame, mgr: TokenManager) -> None:
    """Build the sidebar: LLM input, run button, output box, and playback controls."""
    sidebar = ctk.CTkFrame(frm, fg_color="#ececec")
    sidebar.grid(column=0, row=0, rowspan=2, sticky="ns", padx=8, pady=(43, 8))

    default_font = ctk.CTkFont()
    mono_font = ctk.CTkFont(**MONOSPACE_FONT)

    input_w, input_h = _char_size(default_font, 30, 3)
    llm_input = ctk.CTkTextbox(sidebar, width=input_w, height=input_h, font=default_font)
    llm_input.grid(column=0, row=0)

    default_input_text_color = llm_input.cget("text_color")
    llm_input.insert("1.0", LLM_INPUT_PLACEHOLDER)
    llm_input.configure(text_color=PLACEHOLDER_TEXT_COLOR)

    def _clear_placeholder(_event: tk.Event | None = None) -> None:
        if llm_input.get("1.0", "end-1c") == LLM_INPUT_PLACEHOLDER:
            llm_input.delete("1.0", "end")
            llm_input.configure(text_color=default_input_text_color)

    def _restore_placeholder(_event: tk.Event | None = None) -> None:
        if not llm_input.get("1.0", "end-1c").strip():
            llm_input.delete("1.0", "end")
            llm_input.insert("1.0", LLM_INPUT_PLACEHOLDER)
            llm_input.configure(text_color=PLACEHOLDER_TEXT_COLOR)

    llm_input.bind("<FocusIn>", _clear_placeholder)
    llm_input.bind("<FocusOut>", _restore_placeholder)

    run_button = ctk.CTkButton(sidebar, text="Run", width=input_w)
    run_button.grid(column=0, row=1, pady=32)

    playback = ctk.CTkFrame(sidebar, fg_color="#ececec")
    playback.grid(column=0, row=3, padx=32)
    label_w, _ = _char_size(default_font, 16)
    current_token_display = ctk.CTkLabel(playback, text="token 0 / 0", width=label_w, anchor="center")
    current_token_display.grid(column=1, row=0, pady=32)

    arrow_w, _ = _char_size(default_font, 4)
    back_button = ctk.CTkButton(playback, text="◀ ", width=arrow_w, state="disabled")
    back_button.grid(column=0, row=1, padx=16)
    back_button.configure(command=mgr.prior_token)

    play_stop_button = ctk.CTkButton(playback, text="▶ Play", width=label_w)
    play_stop_button.grid(column=1, row=1, padx=16)
    play_stop_button.configure(command=mgr.toggle_play)

    forward_button = ctk.CTkButton(playback, text="▶ ", width=arrow_w, state="disabled")
    forward_button.grid(column=2, row=1, padx=16)
    forward_button.configure(command=mgr.next_token)

    mgr.tk_token_status = current_token_display
    mgr.tk_toggle_play = play_stop_button
    mgr.tk_forward = forward_button
    mgr.tk_backward = back_button

    run_button.configure(command=lambda: run_llm(sidebar, mgr, llm_input, run_button))

    out_w, out_h = _char_size(mono_font, 30, 16)
    llm_output = ctk.CTkTextbox(sidebar, width=out_w, height=out_h, fg_color="#fff", font=mono_font)
    llm_output.grid(column=0, row=2)
    llm_output.configure(state=ctk.DISABLED)
    mgr.tk_llm_output = llm_output


def run_llm(sidebar: ctk.CTkFrame, mgr: TokenManager, llm_input: ctk.CTkTextbox, run_button: ctk.CTkButton) -> None:
    """Run the LLM on the current input text in a background thread, showing a progress popup."""
    input_text = llm_input.get("1.0", "end-1c").strip() or LLM_INPUT_PLACEHOLDER
    run_button.configure(state="disabled")

    popup = ctk.CTkToplevel(sidebar)
    popup.title("Running")
    popup.resizable(False, False)
    popup.protocol("WM_DELETE_WINDOW", lambda: None)  # block closing mid-run
    popup.transient(mgr.root)

    ctk.CTkLabel(
        popup,
        text="Generating local LLM response - model: SmolLM-135M-Instruct",
    ).grid(row=0, column=0, padx=24, pady=(16, 8))
    progress = ctk.CTkProgressBar(popup, mode="indeterminate", indeterminate_speed=1.2, width=220)
    progress.grid(row=1, column=0, padx=24, pady=(0, 16))
    progress.start()

    popup.update_idletasks()  # force rerender

    x = mgr.root.winfo_rootx() + (mgr.root.winfo_width() - popup.winfo_width()) // 2
    y = mgr.root.winfo_rooty() + (mgr.root.winfo_height() - popup.winfo_height()) // 2
    popup.geometry(f"+{x}+{y}")

    # prevent interaction with main window while llm running
    popup.grab_set()

    result: dict = {}

    def worker() -> None:
        """
        Runs LLM process in background. PyTorch computation does not
        lock the GIL, so running this process as a seperate thread
        serves to keep the GUI process running while waiting.
        """
        if USE_MOCK_LLM:
            result["tokens"] = [ProfiledToken(text, tensors) for text, tensors in TOKENS]
        else:
            llm = ProfiledSmolLM()
            result["tokens"] = llm.run(input_text)

    def check_done(thread: threading.Thread) -> None:
        """Poll the worker thread and, once finished, close the popup and load its results."""
        if thread.is_alive():
            mgr.root.after(100, lambda: check_done(thread))
            return

        progress.stop()
        popup.grab_release()
        popup.destroy()
        run_button.configure(state="normal")
        mgr.set_tokens(result["tokens"])

    """
    Note: the worker thread is daemonized to prevent zombie
    processes sticking around if the GUI is closed mid-inference.
    In the event of the window closing, however, the worker thread
    will not be closed gracefully and may not properly release file locks.
    """

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    check_done(thread)


def create_display(frm: ctk.CTkFrame, mgr: TokenManager) -> None:
    """Build the main display: activation graph and threshold selector."""

    display = ctk.CTkFrame(frm, fg_color="transparent")
    display.grid(column=1, row=0, sticky="nsew")
    display.rowconfigure(0, weight=0)
    display.rowconfigure(1, weight=1)

    activation_f = ctk.CTkFrame(display, fg_color="transparent")
    activation_f.grid(column=0, row=0, sticky="ne")

    activation_l = ctk.CTkLabel(activation_f, text="Activation threshold: ")
    activation_l.grid(row=0, column=0)

    def cb_select(value: str) -> None:
        """Update the graph's activation threshold when a new value is selected."""
        mgr.graph_config.threshold = float(value)
        mgr.render_token()

    default_font = ctk.CTkFont()
    combo_w, _ = _char_size(default_font, 5)
    activation = ctk.CTkComboBox(
        activation_f,
        width=combo_w + 40,  # plus room for the dropdown arrow button
        values=["0.25", "0.33", "0.50", "0.66", "0.75"],
        command=cb_select,
    )
    # CTkComboBox.set() only bypasses the entry's own state check for "readonly", not
    # "disabled" -- setting the starting value has to happen before disabling, or it's a no-op.
    activation.set("0.33")
    activation.configure(state=ctk.DISABLED)
    mgr.tk_cb_activation = activation
    activation.grid(padx=10, pady=1, column=1, row=0, sticky="e")

    text = get_default_graph_text()
    mono_font = ctk.CTkFont(**MONOSPACE_FONT)
    graph_w, graph_h = _char_size(mono_font, mgr.graph_config.x_size * 3 + 2, mgr.graph_config.y_size + 2)
    weights = ctk.CTkTextbox(
        display,
        fg_color="#fff",
        font=mono_font,
        width=graph_w,
        height=graph_h,
    )
    weights.insert("1.0", text)
    weights.configure(state=ctk.DISABLED)
    weights.grid(column=0, row=1)

    mgr.tk_graph = weights


if __name__ == "__main__":
    main()
