from matplotlib.animation import PillowWriter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def animate_eigen(X, M, save_path, b_dim=0, d_dim=0, step_size=100):
    signal = X[b_dim].squeeze().cpu()

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10,5))
    animated_plot = ax1.scatter([], [], c=[], cmap='viridis', s=40)
    animated_signal, = ax2.plot([], [], lw=2)

    ax1.set_xlim([-1.2,1.2])
    ax1.set_ylim([-1.2,1.2])
    ax2.set_xlim([0, len(X)])
    ax2.set_ylim([signal.min(), signal.max()])

    theta = np.linspace(0, 2*np.pi, 400)
    ax1.plot(np.cos(theta), np.sin(theta), linestyle=':')

    def update_data(frame):
        points = M[b_dim, frame*step_size, d_dim, :]
        points_array = np.stack(
            (points.real.cpu().numpy(), points.imag.cpu().numpy()),
            axis=-1
        ).reshape(-1,2)
        animated_plot.set_offsets(points_array)

        colors = np.arange(len(points_array))
        animated_plot.set_array(colors)
        ax1.set_title(f"t = {frame*step_size}")

        t = np.linspace(0, 16000, len(signal))
        animated_signal.set_data(t[:frame*step_size], signal[:frame*step_size])

        return animated_plot, animated_signal

    animation = FuncAnimation(
        fig,
        update_data,
        frames=int(M.shape[1] / step_size),
        interval=step_size,
        repeat=False
    )
    
    animation.save(save_path, writer=PillowWriter(fps=10))
    print('Animation saved!')


from matplotlib.animation import PillowWriter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_eigen_text(
    X, 
    M, 
    save_path, 
    b_dim=0, 
    tokens=None,              # NEW: token ids
    tokenizer=None,           # NEW: tokenizer
    d_dim=0, 
    step_size=100,
    fps=10,                   # NEW: frame rate control
    show_words=True           # NEW: toggle labels
):
    signal = X[b_dim].squeeze().cpu()

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10,5))
    animated_plot = ax1.scatter([], [], c=[], cmap='viridis', s=40)
    animated_signal, = ax2.plot([], [], lw=2)

    text_annotations = []  # store word labels

    ax1.set_xlim([-1.2,1.2])
    ax1.set_ylim([-1.2,1.2])
    ax2.set_xlim([0, len(signal)])
    ax2.set_ylim([signal.min(), signal.max()])

    theta = np.linspace(0, 2*np.pi, 400)
    ax1.plot(np.cos(theta), np.sin(theta), linestyle=':')

    def update_data(frame):
        nonlocal text_annotations

        # Clear previous text
        for txt in text_annotations:
            txt.remove()
        text_annotations = []

        points = M[b_dim, frame*step_size, d_dim, :]
        points_array = np.stack(
            (points.real.cpu().numpy(), points.imag.cpu().numpy()),
            axis=-1
        ).reshape(-1,2)

        animated_plot.set_offsets(points_array)

        colors = np.arange(len(points_array))
        animated_plot.set_array(colors)

        ax1.set_title(f"t = {frame*step_size}")

        # ===== TEXT DECODING PART =====
        if show_words and tokenizer is not None and tokens is not None:
            token_ids = tokens[b_dim][:len(points_array)]

            words = tokenizer.convert_ids_to_tokens(
                token_ids.cpu().numpy().tolist()
            )

            # clean GPT2 artifacts (Ġ = space)
            words = [w.replace("Ġ", " ") for w in words]

            for (x, y), word in zip(points_array, words):
                txt = ax1.text(x, y, word, fontsize=8)
                text_annotations.append(txt)

        # ===== SIGNAL PLOT =====
        t = np.arange(len(signal))
        animated_signal.set_data(
            t[:frame*step_size], 
            signal[:frame*step_size]
        )

        return animated_plot, animated_signal

    animation = FuncAnimation(
        fig,
        update_data,
        frames=int(M.shape[1] / step_size),
        interval=1000 / fps,   # smoother control
        repeat=False
    )

    animation.save(save_path, writer=PillowWriter(fps=fps))
    print('Animation saved!')