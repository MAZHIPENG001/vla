"""桌面主窗口：Task / Episode / Frame 导航、后台读取、播放和同步显示。"""

from concurrent.futures import ThreadPoolExecutor
import json
from queue import Empty, Queue
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .dataset import json_value
from .charts import TrajectoryPlot, StatisticsWindow, scroll_tree


class LeRobotViewer:
    def __init__(self, root, model):
        self.root, self.model = root, model
        # 固定为常规桌面字号，避免远程桌面报告的超高 DPI 挤占图表区域。
        root.tk.call("tk", "scaling", 4 / 3)
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkTooltipFont"):
            tkfont.nametofont(name, root=root).configure(family="Noto Sans CJK SC", size=10)
        style = ttk.Style(root)
        style.configure("Treeview", rowheight=25)
        self.episode = None
        self.frame_index = 0
        self.desired_frame = 0
        self.playing = False
        self.play_after = None
        self.seek_after = None
        self.generation = 0
        self.frame_token = 0
        self.pending = {}
        self.events = Queue()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lerobot-read")
        self.closed = False
        self.photos = {}
        self.camera_arrays = {}
        self.task_rows = model.task_stats
        self.task_lookup = {str(i): item for i, item in enumerate(self.task_rows)}
        self.current_task = None
        self.dim_vars = {}
        self.dim_signature = None
        root.title(f"LeRobot Dataset Viewer · {model.loader.path.name}")
        root.geometry(f"{min(1700, root.winfo_screenwidth()-60)}x{min(1080, root.winfo_screenheight()-80)}")
        root.minsize(1100, 760)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self.root.bind("<Left>", lambda e: self._key_step(e, -1))
        self.root.bind("<Right>", lambda e: self._key_step(e, 1))
        self.root.bind("<space>", self._key_play)
        self.poll_after = root.after(20, self._poll)
        self._filter_tasks()
        nonempty = next((key for key, row in self.task_lookup.items() if row.episode_ids), None)
        if nonempty:
            self.task_tree.selection_set(nonempty)
            self.task_tree.see(nonempty)

    def _build(self):
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text="LeRobot / VLA", font=("sans-serif", 16, "bold")).pack(side="left")
        loader = self.model.loader
        ttk.Label(top, text=f"   {len(loader.tasks):,} Tasks   /   {loader.num_episodes:,} Episodes"
                  f"   /   {len(loader):,} Frames   /   {loader.fps} FPS").pack(side="left")
        ttk.Button(top, text="Task 统计 / 分布 / 导出", command=self.open_statistics).pack(side="right")
        ttk.Label(self.root, text=str(loader.path), padding=(12, 0)).pack(fill="x")
        ttk.Label(self.root, text=self.model.schema.summary(), padding=(12, 0)).pack(fill="x")
        self.status = tk.StringVar(value="准备就绪；相机画面可双击放大")
        ttk.Label(self.root, textvariable=self.status, relief="sunken", padding=(8, 3)).pack(side="bottom", fill="x")
        panes = ttk.Panedwindow(self.root, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=5)
        side = ttk.Frame(panes, width=365)
        content = ttk.Frame(panes)
        panes.add(side, weight=0)
        panes.add(content, weight=1)
        ttk.Label(side, text="1  Task", font=("sans-serif", 11, "bold")).pack(anchor="w")
        self.search = tk.StringVar()
        entry = ttk.Entry(side, textvariable=self.search)
        entry.pack(fill="x", pady=4)
        self.search.trace_add("write", lambda *_: self._filter_tasks())
        ttk.Label(side, text="输入任务名称筛选；完整名称显示在下方").pack(anchor="w")
        frame, self.task_tree = scroll_tree(side, [
            ("name", "Task", 235), ("count", "轨迹", 48), ("frames", "帧数", 65)], height=8)
        frame.pack(fill="both", expand=True)
        self.task_tree.bind("<<TreeviewSelect>>", self._select_task)
        self.task_text = tk.StringVar(value="请选择 Task")
        ttk.Label(side, textvariable=self.task_text, wraplength=350, padding=(2, 5)).pack(fill="x")
        ttk.Label(side, text="2  Episode", font=("sans-serif", 11, "bold")).pack(anchor="w")
        jump = ttk.Frame(side)
        jump.pack(fill="x", pady=4)
        self.episode_input = tk.StringVar()
        ttk.Label(jump, text="episode_index").pack(side="left")
        field = ttk.Entry(jump, textvariable=self.episode_input, width=9)
        field.pack(side="left", padx=4)
        field.bind("<Return>", lambda _: self._jump_episode())
        ttk.Button(jump, text="定位", command=self._jump_episode).pack(side="left")
        frame, self.episode_tree = scroll_tree(side, [
            ("id", "Episode", 100), ("length", "帧数", 65), ("duration", "时长(s)", 85)], height=7)
        frame.pack(fill="both", expand=True)
        self.episode_tree.bind("<<TreeviewSelect>>", self._select_episode)
        nav = ttk.Frame(side)
        nav.pack(fill="x", pady=4)
        ttk.Button(nav, text="◀ 上一轨迹", command=lambda: self._step_episode(-1)).pack(side="left", expand=True)
        ttk.Button(nav, text="下一轨迹 ▶", command=lambda: self._step_episode(1)).pack(side="right", expand=True)
        self.episode_text = tk.StringVar()
        ttk.Label(side, textvariable=self.episode_text, wraplength=350, padding=(2, 5)).pack(fill="x")
        ttk.Button(side, text="查看完整 Episode 元数据", command=self._metadata).pack(fill="x")
        self.frame_title = tk.StringVar(value="3  Frame")
        ttk.Label(content, textvariable=self.frame_title, font=("sans-serif", 11, "bold")).pack(anchor="w")
        cameras = ttk.Frame(content)
        cameras.pack(fill="x")
        self.camera_labels = {}
        for column, key in enumerate(self.model.camera_keys):
            cell = ttk.LabelFrame(cameras, text=key.rsplit(".", 1)[-1], padding=2)
            cell.grid(row=column // 3, column=column % 3, sticky="nsew", padx=2)
            cameras.columnconfigure(column % 3, weight=1, uniform="camera")
            label = ttk.Label(cell, text="等待轨迹", anchor="center", background="#18222d")
            label.pack(fill="both", expand=True)
            label.bind("<Double-Button-1>", lambda _, k=key: self._enlarge_camera(k))
            self.camera_labels[key] = label
        if not self.camera_labels:
            ttk.Label(cameras, text="没有内嵌 RGB 图像，可继续检查数值数据").pack(fill="x")
        if self.model.video_keys:
            ttk.Label(content, text="当前读取器不支持外部视频：" + ", ".join(self.model.video_keys)).pack(fill="x")
        transport = ttk.Frame(content)
        transport.pack(fill="x", pady=5)
        ttk.Button(transport, text="◀", width=3, command=lambda: self.step_frame(-1)).pack(side="left")
        self.play_button = ttk.Button(transport, text="播放", width=6, command=self.toggle_play)
        self.play_button.pack(side="left", padx=2)
        ttk.Button(transport, text="▶", width=3, command=lambda: self.step_frame(1)).pack(side="left")
        self.slider_var = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(transport, from_=0, to=1, variable=self.slider_var, command=self._slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=6)
        self.frame_input = tk.StringVar(value="0")
        self.frame_spin = ttk.Spinbox(transport, from_=0, to=0, width=6,
                                      textvariable=self.frame_input, command=self._jump_frame)
        self.frame_spin.pack(side="left")
        self.frame_spin.bind("<Return>", lambda _: self._jump_frame())
        self.speed = tk.StringVar(value="1×")
        ttk.Combobox(transport, textvariable=self.speed, values=["0.25×", "0.5×", "1×", "2×"],
                     width=6, state="readonly").pack(side="left", padx=5)
        self.loop = tk.BooleanVar(value=False)
        ttk.Checkbutton(transport, text="循环", variable=self.loop).pack(side="left")
        lower = ttk.Panedwindow(content, orient="horizontal")
        lower.pack(fill="both", expand=True)
        chart = ttk.Frame(lower)
        details = ttk.Frame(lower, width=355)
        lower.add(chart, weight=3)
        lower.add(details, weight=1)
        self.dimension_panel = ttk.Frame(chart)
        self.dimension_panel.pack(fill="x")
        self.plot = TrajectoryPlot(chart, self._seek_from_plot)
        self.plot.pack(fill="both", expand=True)
        ttk.Label(details, text="当前帧：全部维度", font=("sans-serif", 10, "bold")).pack(anchor="w")
        frame, self.value_tree = scroll_tree(details, [
            ("dim", "维度", 150), ("state", "State", 78), ("action", "Action", 78),
            ("delta", "Action-State", 92)], height=15)
        frame.pack(fill="both", expand=True)
        ttk.Label(details, text="当前帧全部数值字段 / Task").pack(anchor="w", pady=3)
        raw_frame = ttk.Frame(details)
        raw_frame.pack(fill="both", expand=True)
        self.raw = tk.Text(raw_frame, width=38, height=10, wrap="word", state="disabled", font=("monospace", 9))
        raw_scroll = ttk.Scrollbar(raw_frame, command=self.raw.yview)
        self.raw.configure(yscrollcommand=raw_scroll.set)
        self.raw.pack(side="left", fill="both", expand=True)
        raw_scroll.pack(side="right", fill="y")

    def _submit(self, kind, token, function, *args):
        previous = self.pending.get(kind)
        if previous:
            previous.cancel()
        future = self.executor.submit(function, *args)
        self.pending[kind] = future
        def done(result):
            if result.cancelled():
                return
            try:
                self.events.put((kind, token, result.result(), None))
            except Exception as exc:
                self.events.put((kind, token, None, exc))
        future.add_done_callback(done)

    def _poll(self):
        if self.closed:
            return
        try:
            while True:
                kind, token, value, error = self.events.get_nowait()
                expected = self.generation if kind == "episode" else (self.generation, self.frame_token)
                if token != expected:
                    continue
                if error is not None:
                    self.stop_play()
                    self.status.set(f"读取失败：{error}")
                    messagebox.showerror("读取失败", str(error), parent=self.root)
                    continue
                if kind == "episode":
                    self._episode_ready(value)
                else:
                    self._frame_ready(*value)
        except Empty:
            pass
        self.poll_after = self.root.after(15, self._poll)

    def _filter_tasks(self):
        query = self.search.get().strip().casefold()
        selected = self.task_tree.selection()
        self.task_tree.delete(*self.task_tree.get_children())
        for key, item in self.task_lookup.items():
            if query in item.name.casefold():
                self.task_tree.insert("", "end", iid=key,
                                      values=(item.name, len(item.episode_ids), item.total_frames))
        if selected and self.task_tree.exists(selected[0]):
            self.task_tree.selection_set(selected[0])

    def _select_task(self, _event=None):
        selection = self.task_tree.selection()
        if not selection:
            return
        task = self.task_lookup[selection[0]]
        if task == self.current_task:
            return
        self.stop_play()
        self.current_task = task
        self.task_text.set(f"{task.name}\n{len(task.episode_ids)} Episodes · {task.total_frames:,} 帧\n"
                           f"平均 {task.mean_frames:.1f} 帧 / {task.mean_seconds:.3f} 秒")
        self.episode_tree.delete(*self.episode_tree.get_children())
        for eid in task.episode_ids:
            length = self.model.loader.episodes[eid]["length"]
            self.episode_tree.insert("", "end", iid=str(eid),
                                     values=(eid, length, f"{length / self.model.loader.fps:.3f}"))
        if task.episode_ids:
            self.episode_tree.selection_set(str(task.episode_ids[0]))
        else:
            self._clear_episode()
            self.episode_text.set("此 Task 没有关联的 Episode")

    def _clear_episode(self):
        self.stop_play()
        self.generation += 1
        self.frame_token += 1
        if self.seek_after:
            self.root.after_cancel(self.seek_after)
            self.seek_after = None
        for future in self.pending.values():
            future.cancel()
        self.episode = None
        self.frame_title.set("3  Frame · 等待轨迹")
        self.plot.clear()
        self.value_tree.delete(*self.value_tree.get_children())
        self._set_raw("")
        self.photos.clear()
        self.camera_arrays = {}
        for label in self.camera_labels.values():
            label.configure(image="", text="等待轨迹")

    def _select_episode(self, _event=None):
        selection = self.episode_tree.selection()
        if not selection:
            return
        eid = int(selection[0])
        if self.episode is not None and eid == self.episode.index:
            return
        self._clear_episode()
        self.episode_input.set(str(eid))
        length = self.model.loader.episodes[eid]["length"]
        self.episode_text.set(f"Episode {eid} · {length} 帧 · {self.model.loader.fps} FPS\n"
                              f"采样时长 {length / self.model.loader.fps:.3f} 秒")
        self.status.set(f"正在读取 Episode {eid} 的数值序列…")
        self._submit("episode", self.generation, self.model.read_episode, eid)

    def _jump_episode(self):
        try:
            index = int(self.episode_input.get())
            if self.current_task is None or index not in self.current_task.episode_ids:
                raise ValueError("请输入当前 Task 下的 episode_index")
            self.episode_tree.selection_set(str(index))
            self.episode_tree.see(str(index))
        except ValueError as exc:
            self.status.set(str(exc))

    def _step_episode(self, step):
        if self.current_task is None or not self.current_task.episode_ids:
            return
        ids = self.current_task.episode_ids
        selection = self.episode_tree.selection()
        current = ids.index(int(selection[0])) if selection else 0
        target = ids[min(max(current + step, 0), len(ids) - 1)]
        self.episode_tree.selection_set(str(target))
        self.episode_tree.see(str(target))

    def _episode_ready(self, episode):
        self.episode = episode
        self.frame_index = self.desired_frame = 0
        self.slider.configure(to=max(0, episode.length - 1))
        self.frame_spin.configure(to=episode.length - 1)
        self._set_slider(0)
        self.frame_input.set("0")
        self._build_dimensions()
        self._replot()
        self.request_frame(0)

    def _build_dimensions(self):
        signature = []
        for role in ("state", "action"):
            field = getattr(self.model.schema, role)
            signature.append((role, field.names))
        if signature == self.dim_signature:
            return
        self.dim_signature = signature
        self.dim_vars = {}
        for widget in self.dimension_panel.winfo_children():
            widget.destroy()
        for key, names in signature:
            field = getattr(self.model.schema, key)
            frame = ttk.LabelFrame(self.dimension_panel, text=f"{key.title()}: {field.key or '未映射'} · 显示维度")
            frame.pack(fill="x", pady=2)
            self.dim_vars[key] = [tk.BooleanVar(value=True) for _ in names]
            for dim, variable in enumerate(self.dim_vars[key]):
                # 编号与右侧全部维度名称对应，避免长关节名挤占曲线区域。
                ttk.Checkbutton(frame, text=str(dim), variable=variable,
                                command=self._replot).grid(row=dim // 14, column=dim % 14, sticky="w")
            ttk.Button(frame, text="全选", width=4,
                       command=lambda k=key: self._all_dimensions(k, True)).grid(row=0, column=14)
            ttk.Button(frame, text="清空", width=4,
                       command=lambda k=key: self._all_dimensions(k, False)).grid(row=0, column=15)

    def _all_dimensions(self, key, selected):
        for variable in self.dim_vars[key]:
            variable.set(selected)
        self._replot()

    def _replot(self):
        if self.episode is None:
            return
        dims = {key: [i for i, value in enumerate(variables) if value.get()]
                for key, variables in self.dim_vars.items()}
        self.plot.set_episode(self.episode, dims["state"], dims["action"])
        self.plot.set_frame(self.frame_index)

    def _set_slider(self, index):
        self.setting_slider = True
        try:
            self.slider_var.set(index)
        finally:
            self.setting_slider = False

    def _slider(self, value):
        if getattr(self, "setting_slider", False) or self.episode is None:
            return
        self.stop_play()
        self.desired_frame = int(round(float(value)))
        if self.seek_after:
            self.root.after_cancel(self.seek_after)
        self.seek_after = self.root.after(35, self._seek_debounced)

    def _seek_debounced(self):
        self.seek_after = None
        self.request_frame(self.desired_frame)

    def _jump_frame(self):
        if self.episode is None:
            return
        try:
            index = int(self.frame_input.get())
            if not 0 <= index < self.episode.length:
                raise ValueError()
            self.stop_play()
            self.request_frame(index)
        except ValueError:
            self.status.set(f"帧索引范围：0 到 {self.episode.length-1}")

    def step_frame(self, step):
        if self.episode is not None:
            self.stop_play()
            self.request_frame(self.desired_frame + step)

    def request_frame(self, index):
        if self.episode is None:
            return
        if self.seek_after:
            self.root.after_cancel(self.seek_after)
            self.seek_after = None
        index = min(max(int(index), 0), self.episode.length - 1)
        self.desired_frame = index
        self.frame_token += 1
        self.status.set(f"Episode {self.episode.index} · 正在读取帧 {index}…")
        self._submit("frame", (self.generation, self.frame_token),
                     self._read_frame_result, self.episode.index, index)

    def _seek_from_plot(self, index):
        self.stop_play()
        self.request_frame(index)

    def _read_frame_result(self, episode, index):
        return index, self.model.read_images(episode, index)

    def _frame_ready(self, index, images):
        started = time.monotonic()
        self.frame_index = index
        self.camera_arrays = images
        self._show_images()
        self._set_slider(index)
        self.frame_input.set(str(index))
        values = self.episode.frame(index)
        recorded_index = values.get("frame_index", index)
        timestamp = float(self.episode.times[index])
        self.frame_title.set(f"3  Frame {index} / {self.episode.length-1}   ·   "
                             f"frame_index={recorded_index}   ·   t={timestamp:.6f} s")
        self.plot.set_frame(index)
        self.value_tree.delete(*self.value_tree.get_children())
        state_series, action_series = self.episode.state, self.episode.action
        state = state_series[index] if state_series is not None else np.array([])
        action = action_series[index] if action_series is not None else np.array([])
        state_names = self.model.schema.state.names
        action_names = self.model.schema.action.names
        for dim in range(max(len(state), len(action))):
            sn = state_names[dim] if dim < len(state_names) else "—"
            an = action_names[dim] if dim < len(action_names) else "—"
            name = sn if sn == an else f"S:{sn} / A:{an}"
            s = f"{state[dim]:.8g}" if dim < len(state) else "—"
            a = f"{action[dim]:.8g}" if dim < len(action) else "—"
            delta = f"{action[dim] - state[dim]:.8g}" if dim < min(len(state), len(action)) else "—"
            self.value_tree.insert("", "end", values=(f"{dim}: {name}", s, a, delta))
        values["camera_info"] = {key: {"shape_CHW": list(image.shape), "dtype": str(image.dtype)}
                                 for key, image in images.items()}
        self._set_raw(json.dumps(json_value(values), indent=2, ensure_ascii=False))
        self.status.set(f"Episode {self.episode.index} · 帧 {index} · t={timestamp:.6f}s · "
                        "图像 / 曲线标记 / 全部数值已同步；相机双击放大")
        if self.playing:
            next_index = index + 1
            if next_index >= self.episode.length:
                if self.loop.get():
                    next_index = 0
                else:
                    self.stop_play()
                    return
            speed = float(self.speed.get().rstrip("×"))
            interval = 1 / (float(self.model.loader.fps) * speed)
            delay = max(1, int(1000 * (interval - (time.monotonic() - started))))
            self.play_after = self.root.after(delay, lambda: self.request_frame(next_index))

    def _show_images(self):
        for key, array in self.camera_arrays.items():
            label = self.camera_labels[key]
            picture = Image.fromarray(array.transpose(1, 2, 0))
            width = max(120, label.winfo_width())
            if width < 150:
                width = max(200, (self.root.winfo_width()-420) // max(1, min(3, len(self.camera_labels))))
            picture.thumbnail((min(width, 480), 195), Image.Resampling.LANCZOS)
            self.photos[key] = ImageTk.PhotoImage(picture, master=self.root)
            label.configure(image=self.photos[key], text="")

    def _set_raw(self, text):
        self.raw.configure(state="normal")
        self.raw.delete("1.0", "end")
        self.raw.insert("1.0", text)
        self.raw.configure(state="disabled")

    def toggle_play(self):
        if self.playing:
            self.stop_play()
        elif self.episode is not None:
            self.playing = True
            self.play_button.configure(text="暂停")
            self.request_frame(0 if self.frame_index == self.episode.length-1 else self.frame_index)

    def stop_play(self):
        self.playing = False
        if self.play_after:
            self.root.after_cancel(self.play_after)
            self.play_after = None
        if hasattr(self, "play_button"):
            self.play_button.configure(text="播放")

    def _key_step(self, event, step):
        if not isinstance(event.widget, (ttk.Entry, ttk.Spinbox, tk.Text)):
            self.step_frame(step)
            return "break"

    def _key_play(self, event):
        if not isinstance(event.widget, (ttk.Entry, ttk.Spinbox, tk.Text)):
            self.toggle_play()
            return "break"

    def _metadata(self):
        if self.episode is None:
            return
        window = tk.Toplevel(self.root)
        window.title(f"Episode {self.episode.index} · 完整信息")
        window.geometry("760x650")
        text = tk.Text(window, wrap="word")
        text.pack(fill="both", expand=True)
        loader = self.model.loader
        payload = {
            "episode": self.episode.metadata,
            "fps": loader.fps,
            "sample_duration_seconds": self.episode.length / loader.fps,
            "first_timestamp": self.episode.times[0],
            "last_timestamp": self.episode.times[-1],
            "timestamp_span_seconds": self.episode.times[-1] - self.episode.times[0],
            "robot_type": loader.info.get("robot_type"),
            "data_path": loader.info["data_path"].format(
                episode_chunk=self.episode.index // loader.info["chunks_size"],
                episode_index=self.episode.index),
            "features": loader.features,
            "field_mapping": {
                role: {"key": field.key, "dimensions": field.width, "names": field.names}
                for role, field in (("state", self.model.schema.state), ("action", self.model.schema.action))
            },
        }
        text.insert("1.0", json.dumps(json_value(payload), indent=2, ensure_ascii=False))
        text.configure(state="disabled")

    def _enlarge_camera(self, key):
        if key not in self.camera_arrays:
            return
        window = tk.Toplevel(self.root)
        window.title(f"{key} · Episode {self.episode.index} · Frame {self.frame_index}（快照）")
        image = Image.fromarray(self.camera_arrays[key].transpose(1, 2, 0))
        photo = ImageTk.PhotoImage(image, master=window)
        label = ttk.Label(window, image=photo)
        label.image = photo
        label.pack()

    def close(self):
        self.closed = True
        self.stop_play()
        self.root.after_cancel(self.poll_after)
        if self.seek_after:
            self.root.after_cancel(self.seek_after)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def open_statistics(self):
        return StatisticsWindow(self.root, self.model.task_stats, self.select_task)

    def select_task(self, name):
        """供统计组件通过任务名导航，组件无需访问主窗口控件。"""
        key = next(key for key, row in self.task_lookup.items() if row.name == name)
        self.search.set("")
        self.task_tree.selection_set(key)
        self.task_tree.see(key)
        self.root.lift()
