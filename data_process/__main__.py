"""
命令行入口：浏览器服务 view 与终端单帧读取 read。
uv run python -m data_process read
uv run python -m data_process view
"""

import argparse
from pathlib import Path
import sys

from .config import DEFAULT_DATA_PATH
from .dataset import LeRobotReader

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LeRobot / VLA 数据读取与浏览器可视化")
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser("view", help="启动本地服务并在浏览器中操作")
    read = commands.add_parser("read", help="在终端预览一帧数据")
    for command in (view, read):
        command.add_argument("--path", type=Path, default=DEFAULT_DATA_PATH, help="本地数据集根目录")
    view.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机；远程访问使用 0.0.0.0")
    view.add_argument("--port", type=int, default=8765, help="HTTP 端口，默认 8765；0 为自动分配")
    view.add_argument("--no-browser", action="store_true", help="仅启动服务，不自动打开浏览器")
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
    from .dataset import BrowserModel
    from .web_server import serve

    if not 0 <= args.port <= 65535:
        parser.error("端口必须在 0–65535 之间")
    try:
        model = BrowserModel(args.path, state_key=args.state_key, action_key=args.action_key)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if args.episode is not None and args.episode not in model.loader.episodes:
        parser.error(f"找不到 Episode {args.episode}")
    try:
        serve(model, host=args.host, port=args.port, episode=args.episode,
              open_browser=not args.no_browser)
    except OSError as exc:
        parser.exit(1, f"无法启动浏览器服务：{exc}\n可使用 --port 指定其他端口。\n")


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
