"""桌面图表组件：轨迹曲线、Task 统计及共用表格。"""

import csv
import os
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# 在导入 Matplotlib 前设置缓存，数据集保持只读。
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vla-matplotlib"))

import numpy as np
from matplotlib import colormaps, rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]


def scroll_tree(parent, columns, height=8):
    frame = ttk.Frame(parent)
    tree = ttk.Treeview(frame, columns=[key for key, _, _ in columns],
                        show="headings", selectmode="browse", height=height)
    for key, label, width in columns:
        tree.heading(key, text=label)
        tree.column(key, width=width, minwidth=45, stretch=key == columns[0][0])
    scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    horizontal = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=scroll.set, xscrollcommand=horizontal.set)
    tree.grid(row=0, column=0, sticky="nsew")
    scroll.grid(row=0, column=1, sticky="ns")
    horizontal.grid(row=1, column=0, sticky="ew")
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)
    return frame, tree


class TrajectoryPlot(ttk.Frame):
    """三幅共享时间轴的图；拖动/缩放由 Matplotlib 工具栏提供。"""

    def __init__(self, parent, on_seek):
        super().__init__(parent)
        self.on_seek = on_seek
        self.figure = Figure(figsize=(9, 6), dpi=90)
        self.axes = self.figure.subplots(3, 1, sharex=True)
        self.figure.subplots_adjust(left=.07, right=.985, top=.95, bottom=.08, hspace=.40)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.hint = ttk.Label(self, text="拖动工具：平移 / 框选缩放；Home：复位；单击曲线：跳帧；悬停：查看数值")
        self.hint.pack(side="bottom", fill="x")
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.episode = None
        self.markers = []
        self.backgrounds = {}
        self.current_frame = 0
        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.mpl_connect("button_press_event", self._click)
        self.canvas.mpl_connect("motion_notify_event", self._hover)

    def set_episode(self, episode, state_dims, action_dims):
        self.episode = episode
        self.markers = []
        self.backgrounds.clear()
        self.lines = []
        colors = colormaps["tab20"]
        for ax in self.axes:
            ax.clear()
            ax.grid(alpha=.22)
            ax.set_ylabel("Value", fontsize=8)
            ax.tick_params(labelsize=8)
        definitions = [
            (self.axes[0], episode.schema.state, episode.state, state_dims, "State"),
            (self.axes[1], episode.schema.action, episode.action, action_dims, "Action"),
        ]
        for ax, field, values, dims, title in definitions:
            ax.set_title(f"{title}: {field.key or 'unmapped'} · complete episode", fontsize=10, loc="left")
            if isinstance(values, np.ndarray) and values.ndim == 2:
                for dim in dims:
                    line, = ax.plot(episode.times, values[:, dim], lw=.9,
                                    color=colors(dim % 20), label=f"{dim}: {field.names[dim]}")
                    self.lines.append((ax, line, values[:, dim]))
                if dims:
                    ax.legend(fontsize=6, ncol=min(7, len(dims)), loc="upper right")
            else:
                ax.text(.5, .5, f"Select field with --{title.lower()}-key", transform=ax.transAxes, ha="center")
        compare = self.axes[2]
        compare.set_title("State / Action · by dimension index (solid / dashed)", fontsize=10, loc="left")
        state, action = episode.state, episode.action
        if (isinstance(state, np.ndarray) and state.ndim == 2
                and isinstance(action, np.ndarray) and action.ndim == 2):
            for dim in sorted(set(state_dims) & set(action_dims)):
                for values, style, prefix in [(state, "-", "S"), (action, "--", "A")]:
                    line, = compare.plot(episode.times, values[:, dim], style,
                                         lw=.9, color=colors(dim % 20), label=f"{prefix}{dim}")
                    self.lines.append((compare, line, values[:, dim]))
        compare.set_xlabel("Recorded timestamp (s)", fontsize=8)
        for ax in self.axes:
            marker = ax.axvline(episode.times[0], color="#d13c4b", lw=1.3,
                               linestyle="--", animated=True)
            self.markers.append((ax, marker))
        self.current_frame = 0
        self.toolbar.update()
        self.canvas.draw()

    def clear(self):
        self.episode = None
        self.markers = []
        self.backgrounds.clear()
        for ax in self.axes:
            ax.clear()
        self.canvas.draw_idle()

    def _on_draw(self, _event):
        self.backgrounds = {ax: self.canvas.copy_from_bbox(ax.bbox) for ax in self.axes}
        self._draw_markers()

    def _draw_markers(self):
        if self.episode is None:
            return
        timestamp = self.episode.times[self.current_frame]
        for ax, marker in self.markers:
            if ax in self.backgrounds:
                self.canvas.restore_region(self.backgrounds[ax])
                marker.set_xdata([timestamp, timestamp])
                ax.draw_artist(marker)
                self.canvas.blit(ax.bbox)

    def set_frame(self, index):
        self.current_frame = index
        self._draw_markers()

    def _nearest(self, timestamp):
        times = self.episode.times
        index = int(np.searchsorted(times, timestamp))
        choices = [i for i in (index - 1, index) if 0 <= i < len(times)]
        return min(choices, key=lambda i: abs(times[i] - timestamp))

    def _click(self, event):
        if self.episode is not None and event.inaxes in self.axes and event.button == 1 and not self.toolbar.mode:
            self.on_seek(self._nearest(event.xdata))

    def _hover(self, event):
        if self.episode is None or event.inaxes not in self.axes or event.xdata is None:
            return
        index = self._nearest(event.xdata)
        lines = [(line, values) for ax, line, values in self.lines if ax == event.inaxes]
        if lines and event.y is not None:
            line, values = min(lines, key=lambda item: abs(
                event.inaxes.transData.transform((event.xdata, item[1][index]))[1] - event.y))
            self.hint.configure(text=f"帧 {index}  ·  t={self.episode.times[index]:.6f}s  ·  "
                                f"{line.get_label()} = {values[index]:.8g}  ·  单击跳帧")


class StatisticsWindow:
    """接收统计数据和选择回调，不依赖主窗口的控件或内部状态。"""

    def __init__(self, parent, task_stats, on_task_select):
        self.task_lookup = {str(i): row for i, row in enumerate(task_stats)}
        self.on_task_select = on_task_select
        self.window = tk.Toplevel(parent)
        self.window.title("Task 数据统计与分布")
        self.window.geometry("1350x870")
        self.query = tk.StringVar()
        self.metric = tk.StringVar(value="总帧数")
        self.sort_key = "frames"
        self.reverse = True
        top = ttk.Frame(self.window, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="筛选 Task：").pack(side="left")
        ttk.Entry(top, textvariable=self.query, width=45).pack(side="left", padx=4)
        ttk.Button(top, text="导出当前统计 CSV", command=self._export).pack(side="right")
        ttk.Label(self.window, text="双击表格或单击柱状图跳转 Task；点击表头排序。"
                  "平均时长 = 平均帧数 / FPS；多任务轨迹分别计入各 Task，汇总可能重复。",
                  padding=(8, 2)).pack(fill="x")
        frame, self.tree = scroll_tree(self.window, [
            ("name", "Task 名称", 730), ("episodes", "Episode 数", 100),
            ("frames", "总帧数", 100), ("mean", "平均帧数", 100),
            ("seconds", "平均时长(s)", 110)], height=11)
        frame.pack(fill="both", expand=True, padx=8)
        for key in ("name", "episodes", "frames", "mean", "seconds"):
            self.tree.heading(key, command=lambda k=key: self._sort(k))
        self.tree.bind("<Double-Button-1>", self._open_task)
        options = ttk.Frame(self.window, padding=8)
        options.pack(fill="x")
        ttk.Label(options, text="分布指标（显示当前筛选结果最多的前 20 个 Task）：").pack(side="left")
        combo = ttk.Combobox(options, textvariable=self.metric, state="readonly",
                             values=["总帧数", "Episode 数", "平均帧数", "平均时长(s)"], width=16)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda _: self.refresh())
        self.figure = Figure(figsize=(12, 4), dpi=90)
        self.ax = self.figure.subplots()
        self.figure.subplots_adjust(left=.40, right=.97, top=.96, bottom=.12)
        self.canvas = FigureCanvasTkAgg(self.figure, self.window)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8)
        self.canvas.mpl_connect("button_press_event", self._bar_click)
        NavigationToolbar2Tk(self.canvas, self.window)
        self.query.trace_add("write", lambda *_: self.refresh())
        self.refresh()

    def _value(self, item, key):
        return {
            "name": item.name, "episodes": len(item.episode_ids), "frames": item.total_frames,
            "mean": item.mean_frames, "seconds": item.mean_seconds,
        }[key]

    def _sort(self, key):
        self.reverse = not self.reverse if key == self.sort_key else key != "name"
        self.sort_key = key
        self.refresh()

    def refresh(self):
        query = self.query.get().casefold().strip()
        self.rows = [(key, row) for key, row in self.task_lookup.items() if query in row.name.casefold()]
        self.rows.sort(key=lambda item: self._value(item[1], self.sort_key), reverse=self.reverse)
        self.tree.delete(*self.tree.get_children())
        for key, row in self.rows:
            self.tree.insert("", "end", iid=key, values=(
                row.name, len(row.episode_ids), row.total_frames,
                f"{row.mean_frames:.2f}", f"{row.mean_seconds:.3f}"))
        metric_key = {"总帧数": "frames", "Episode 数": "episodes", "平均帧数": "mean",
                      "平均时长(s)": "seconds"}[self.metric.get()]
        self.chart_rows = sorted(self.rows, key=lambda item: self._value(item[1], metric_key), reverse=True)[:20]
        self.ax.clear()
        positions = np.arange(len(self.chart_rows))
        self.ax.barh(positions, [self._value(row, metric_key) for _, row in self.chart_rows], color="#478abe")
        self.ax.set_yticks(positions, [row.name[:60] + ("…" if len(row.name) > 60 else "")
                                      for _, row in self.chart_rows], fontsize=7)
        self.ax.invert_yaxis()
        self.ax.set_xlabel({"frames": "Total frames", "episodes": "Episode count",
                            "mean": "Mean frames", "seconds": "Mean duration (s)"}[metric_key])
        self.ax.grid(axis="x", alpha=.2)
        self.canvas.draw_idle()

    def _navigate(self, key):
        self.on_task_select(self.task_lookup[key].name)

    def _open_task(self, _event):
        selection = self.tree.selection()
        if selection:
            self._navigate(selection[0])

    def _bar_click(self, event):
        if event.inaxes == self.ax and event.ydata is not None and not self.canvas.toolbar.mode:
            index = int(round(event.ydata))
            if 0 <= index < len(self.chart_rows):
                self._navigate(self.chart_rows[index][0])

    def _export(self):
        target = filedialog.asksaveasfilename(parent=self.window, defaultextension=".csv",
                                            initialfile="lerobot_task_statistics.csv",
                                            filetypes=[("CSV", "*.csv")])
        if target:
            try:
                with open(target, "w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["Task", "Episode 数", "总帧数", "平均帧数", "平均时长(s)"])
                    for _, row in self.rows:
                        writer.writerow([row.name, len(row.episode_ids), row.total_frames,
                                         row.mean_frames, row.mean_seconds])
            except OSError as exc:
                messagebox.showerror("导出失败", str(exc), parent=self.window)
