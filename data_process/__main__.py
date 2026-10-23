'''
统一命令行入口：
python -m data_process view
python -m data_process read
'''

import argparse
from pathlib import Path
import sys

from .dataset import LeRobotReader
from .config import DEFAULT_DATA_PATH

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LeRobot / VLA 数据读取与桌面可视化")
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser("view", help="打开独立桌面窗口")
    read = commands.add_parser("read", help="在终端预览一帧数据")
    for command in (view, read):
        command.add_argument("--path", type=Path, default=DEFAULT_DATA_PATH, help="本地数据集根目录")
    view.add_argument("--episode", type=int, help="启动时定位的 episode_index")
    view.add_argument("--state-key", help="State 字段名；默认根据 features 自动识别")
    view.add_argument("--action-key", help="Action 字段名；默认根据 features 自动识别")
    read.add_argument("--episode", type=int, default=0, help="轨迹编号")
    read.add_argument("--frame", type=int, default=0, help="轨迹内帧索引")
    read.add_argument("--columns", nargs="+", help="仅读取指定字段")
    return parser


def _read(args) -> None:
    import numpy as np

    loader = LeRobotReader(args.path)
    print(f"\33[92m数据集: {loader.path}\33[0m")
    print(f"\33[92m轨迹数: {loader.num_episodes}, 总帧数: {len(loader)}, FPS: {loader.fps}\33[0m")
    print(f"\33[92m相机: {loader.camera_keys}\33[0m")
    
    frame = loader.read_frame(args.episode, args.frame, args.columns)
    print(f"\33[93m轨迹 {args.episode}, 帧 {args.frame}:\33[0m")
    for key, value in frame.items():
        if isinstance(value, np.ndarray):
            print(f"\33[94m  {key}: shape={value.shape}, dtype={value.dtype}\33[0m")
        else:
            print(f"\33[95m  {key}: {value}\33[0m")


def _view(args, parser) -> None:
    import tkinter as tk
    from tkinter import ttk

    from .dataset import BrowserModel
    from .viewer import LeRobotViewer

    try:
        model = BrowserModel(args.path, state_key=args.state_key, action_key=args.action_key)
    except ValueError as exc:
        parser.error(str(exc))
    if args.episode is not None and args.episode not in model.loader.episodes:
        parser.error(f"找不到 Episode {args.episode}")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        parser.exit(1, f"无法创建桌面窗口：{exc}\n请在有桌面显示服务的会话中运行。\n")
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    viewer = LeRobotViewer(root, model)
    if args.episode is not None:
        key = next(key for key, row in viewer.task_lookup.items() if args.episode in row.episode_ids)
        viewer.task_tree.selection_set(key)
        viewer.task_tree.see(key)
        viewer._select_task()
        viewer.episode_tree.selection_set(str(args.episode))
    root.mainloop()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments or ["view"])
    if args.command == "read":
        _read(args)
    else:
        _view(args, parser)


if __name__ == "__main__":
    main()
