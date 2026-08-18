import itertools
import os
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from logging import getLogger
from tkinter import ttk

from activation_viz.fixtures import TOKENS, get_default_graph_text
from activation_viz.llm import ProfiledSmolLM, ProfiledToken, download_model

USE_MOCK_LLM = os.environ.get("USE_MOCK_LLM", "").lower() in ("1", "true", "yes")

logger = getLogger(__name__)


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
        root: tk.Tk,
        graph_config: GraphConfig,
    ) -> None:
        """Initialize GUI state for the given Tk root and graph configuration."""
        self.root = root
        self.graph_config = graph_config
        self.curr_token: int = 0
        self.last_token: int = 0
        self.is_playing = False
        self.tokens: list[ProfiledToken] = []

        self.tk_graph: tk.Text
        self.tk_token_status: ttk.Label
        self.tk_cb_activation: ttk.Combobox
        self.tk_llm_output: tk.Text
        self.tk_toggle_play: ttk.Button
        self.tk_forward: ttk.Button
        self.tk_backward: ttk.Button

    def set_tokens(self, tokens: list[ProfiledToken]) -> None:
        """Load a new list of tokens and start playback from the first token."""
        assert len(tokens) > 0
        assert self.tk_cb_activation

        self.curr_token = 0
        self.last_token = len(tokens) - 1
        self.tokens = tokens

        self.tk_cb_activation.config(state=tk.NORMAL)

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
            return ValueError("tk_graph must be set before using render_token")
        if self.tk_llm_output is None:
            return ValueError("tk_llm_output must be set before using render_token")
        if self.tk_token_status is None:
            return ValueError("tk_token_status must be set before using render_token")

        # Render tensors on graph
        compressed_tensors = self._compress_tensors()
        graph_text: str = self._graph_weights(compressed_tensors)
        self.tk_graph.config(state=tk.NORMAL)
        self.tk_graph.delete("1.0", "end")
        self.tk_graph.insert("1.0", graph_text)
        self.tk_graph.config(state=tk.DISABLED)

        # Render llm output box
        llm_text_tokens = [t.text for t in self.tokens[0 : self.curr_token + 1]]
        llm_output = "".join(llm_text_tokens)
        self.tk_llm_output.config(state=tk.NORMAL)
        self.tk_llm_output.delete("1.0", "end")
        self.tk_llm_output.insert("1.0", llm_output)
        self.tk_llm_output.config(state=tk.DISABLED)

        # Render token x / x label
        self.tk_token_status.config(text=f"token {self.curr_token} / {self.last_token}")

    def toggle_play(self) -> None:
        """Toggle between playing and paused playback state."""
        if self.is_playing:
            # Switch state to paused
            self.is_playing = False
            self.tk_toggle_play.config(text="▶ Play")
            self.tk_backward.config(state="normal")
            self.tk_forward.config(state="normal")
            return

        self.is_playing = True
        self.tk_toggle_play.config(text="⏸ Pause")
        self.tk_backward.config(state="disabled")
        self.tk_forward.config(state="disabled")

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

    root = tk.Tk()
    main = ttk.Frame(root, padding=10)
    main.grid(sticky="nsew")
    main.columnconfigure(0, weight=0)
    main.columnconfigure(1, weight=1)

    graph_config = GraphConfig(30, 32, 0.33)
    mgr = TokenManager(root, graph_config)

    create_display(main, mgr)
    create_sidebar(main, mgr)

    main.mainloop()


def create_sidebar(frm: ttk.Frame, mgr: TokenManager) -> None:
    """Build the sidebar: LLM input, run button, output box, and playback controls."""
    sidebar_style = ttk.Style()
    llm_input_style_a = ttk.Style()
    llm_input_style_b = ttk.Style()
    grey_label = ttk.Style()

    sidebar_style.configure("SideBar.TFrame", background="#ececec")
    llm_input_style_a.configure("InputA.TEntry", padding=(8, 8, 8, 200), foreground="grey")
    llm_input_style_b.configure("InputB.TEntry", padding=(8, 8, 8, 200), foreground="black")
    grey_label.configure("Grey.Label", background="#ececec")

    sidebar = ttk.Frame(frm, style="SideBar.TFrame")
    sidebar.grid(column=0, row=0, rowspan=2, sticky="ns", padx=8, pady=(43, 8))

    llm_input = ttk.Entry(sidebar, width=30, style="InputA.TEntry")
    llm_input.grid(column=0, row=0)
    llm_input_placeholder = "What is the capital of France?"
    llm_input.insert(0, llm_input_placeholder)

    def input_focus_in(event: tk.Event) -> None:
        """Clear the placeholder text when the input gains focus."""
        del event  # unused
        # check for current style to prevent deleting input when user enters placeholder verbatim
        if llm_input.cget("style") == "InputA.TEntry" and llm_input.get() == llm_input_placeholder:
            llm_input.delete(0, tk.END)
            llm_input.config(style="InputB.TEntry")

    def input_focus_out(event: tk.Event) -> None:
        """Restore the placeholder text if the input is left empty."""
        del event  # unused
        if llm_input.get() == "":
            llm_input.insert(0, llm_input_placeholder)
            llm_input.config(style="InputA.TEntry")

    llm_input.bind("<FocusIn>", input_focus_in)
    llm_input.bind("<FocusOut>", input_focus_out)

    run_button = ttk.Button(sidebar, text="Run", width=30)
    run_button.grid(column=0, row=1, pady=32)

    playback = ttk.Frame(sidebar, style="SideBar.TFrame")
    playback.grid(column=0, row=3, padx=32)
    current_token_display = ttk.Label(playback, text="token 0 / 0", width=16, style="Grey.Label", anchor="center")
    current_token_display.grid(column=1, row=0, pady=32)

    back_button = ttk.Button(playback, text="◀ ", width=4, state="disabled")
    back_button.grid(column=0, row=1, padx=16)
    back_button.config(command=mgr.prior_token)

    play_stop_button = ttk.Button(playback, text="▶ Play", width=16)
    play_stop_button.grid(column=1, row=1, padx=16)
    play_stop_button.config(command=mgr.toggle_play)

    forward_button = ttk.Button(playback, text="▶ ", width=4, state="disabled")
    forward_button.grid(column=2, row=1, padx=16)
    forward_button.config(command=mgr.next_token)

    mgr.tk_token_status = current_token_display
    mgr.tk_toggle_play = play_stop_button
    mgr.tk_forward = forward_button
    mgr.tk_backward = back_button

    run_button.config(command=lambda: run_llm(sidebar, mgr, llm_input, run_button))

    llm_output = tk.Text(sidebar, width=30, height=16, background="#fff")
    llm_output.grid(column=0, row=2)
    llm_output.config(state=tk.DISABLED)
    mgr.tk_llm_output = llm_output


def run_llm(sidebar: ttk.Frame, mgr: TokenManager, llm_input: ttk.Entry, run_button: ttk.Button) -> None:
    """Run the LLM on the current input text in a background thread, showing a progress popup."""
    input_text = llm_input.get()
    run_button.config(state="disabled")

    popup = tk.Toplevel(sidebar)
    popup.title("Running")
    popup.resizable(False, False)
    popup.protocol("WM_DELETE_WINDOW", lambda: None)  # block closing mid-run
    popup.transient(mgr.root)

    ttk.Label(
        popup,
        text="Generating local LLM response - model: SmolLM-135M-Instruct",
        padding=(24, 16, 24, 8),
    ).grid(row=0, column=0)
    progress = ttk.Progressbar(popup, mode="indeterminate", length=220)
    progress.grid(row=1, column=0, padx=24, pady=(0, 16))
    progress.start(10)

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
        run_button.config(state="normal")
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


def create_display(frm: ttk.Frame, mgr: TokenManager) -> None:
    """Build the main display: activation graph and threshold selector."""

    display = ttk.Frame(frm)
    display.grid(column=1, row=0, sticky="nsew")
    display.rowconfigure(0, weight=0)
    display.rowconfigure(1, weight=1)

    activation_f = ttk.Frame(display)
    activation_f.grid(column=0, row=0, sticky="ne")

    activation_l = ttk.Label(activation_f, text="Activation threshold: ")
    activation_l.grid(row=0, column=0)

    activation = ttk.Combobox(activation_f, width=5, values=["0.25", "0.33", "0.50", "0.66", "0.75"])
    activation.config(state=tk.DISABLED)
    mgr.tk_cb_activation = activation

    def cb_select(event: tk.Event) -> None:
        """Update the graph's activation threshold when a new value is selected."""
        del event  # unused
        mgr.graph_config.threshold = float(activation.get())
        mgr.render_token()
        activation.selection_clear()

    activation.bind("<<ComboboxSelected>>", func=cb_select)
    activation.current(1)
    activation.state(["readonly"])
    activation.grid(padx=10, pady=1, column=1, row=0, sticky="e")

    weights = tk.Text(display, bg="#fff", width=72, height=33)
    text = get_default_graph_text()
    weights = tk.Text(
        display,
        bg="#fff",
        width=mgr.graph_config.x_size * 3,
        height=mgr.graph_config.y_size + 2,
    )
    weights.insert("1.0", text)
    weights.config(state=tk.DISABLED)
    weights.grid(column=0, row=1)

    mgr.tk_graph = weights


if __name__ == "__main__":
    main()
