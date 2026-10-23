"""HTTP 路由、静态资源与本地服务生命周期；业务逻辑位于 application.py。"""

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
from threading import Timer
from urllib.parse import parse_qs, urlsplit
import webbrowser

from .application import BrowserApplication, clean_json

WEB_DIRECTORY = Path(__file__).parent / "web"
STATIC_FILES = {
    "/": ("index.html", "text/html"),
    "/style.css": ("style.css", "text/css"),
    **{f"/{name}.js": (f"{name}.js", "text/javascript")
       for name in ("app", "common", "chart", "statistics", "quality")},
}


class BrowserServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app):
        self.app = app
        super().__init__(address, BrowserHandler)


class BrowserHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_content(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, value, status=200):
        self.send_content(json.dumps(clean_json(value), ensure_ascii=False, allow_nan=False).encode(),
                          "application/json; charset=utf-8", status)

    def _check_host(self):
        host = urlsplit("//" + self.headers.get("Host", "")).hostname
        allowed = {"localhost", "127.0.0.1", self.server.server_address[0]}
        if self.server.server_address[0] == "0.0.0.0":
            try:
                ipaddress.ip_address(host or "")
                return
            except ValueError:
                pass
        if host not in allowed:
            raise PermissionError("Host 不被允许")

    def _dispatch(self, post=False):
        self._check_host()
        app = self.server.app
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        if post:
            if self.headers.get("X-Data-Process-Token") != app.token:
                raise PermissionError("请求令牌无效，请刷新页面")
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                raise PermissionError("只允许同源操作")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4 * 1024 * 1024:
                raise ValueError("请求长度无效")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("请求必须为 JSON 对象")
            if url.path == "/api/quality/start":
                self.send_json(app.start_quality(body), HTTPStatus.ACCEPTED)
            elif url.path == "/api/quality/stop":
                self.send_json(app.stop_quality(body.get("id")))
            else:
                raise KeyError("接口不存在")
        elif url.path == "/api/info":
            self.send_json(app.info())
        elif url.path == "/api/episode":
            self.send_json(app.episode_data(int(query["episode"][0])))
        elif url.path == "/api/frame":
            self.send_json(app.frame_data(int(query["episode"][0]), int(query["frame"][0])))
        elif url.path == "/api/image":
            self.send_content(app.image_data(int(query["episode"][0]), int(query["frame"][0]), query["camera"][0]), "image/png")
        elif url.path == "/api/quality/status":
            self.send_json(app.quality_status(query["id"][0]))
        elif url.path in STATIC_FILES:
            name, mime = STATIC_FILES[url.path]
            content = (WEB_DIRECTORY / name).read_bytes()
            if name == "index.html":
                content = content.replace(b"__API_TOKEN__", app.token.encode())
            self.send_content(content, mime + "; charset=utf-8")
        else:
            raise KeyError("页面不存在")

    def _handle(self, post=False):
        try:
            self._dispatch(post)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except PermissionError as exc:
            self.send_json(dict(error=str(exc)), 403)
        except (KeyError, FileNotFoundError) as exc:
            self.send_json(dict(error=str(exc)), 404)
        except (ValueError, TypeError, IndexError, NotImplementedError) as exc:
            self.send_json(dict(error=str(exc)), 400)
        except Exception as exc:
            self.send_json(dict(error=f"读取失败：{exc}"), 500)

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle(True)


def serve(model, *, host="127.0.0.1", port=8765, episode=None, open_browser=True):
    app = BrowserApplication(model, episode)
    try:
        server = BrowserServer((host, port), app)
    except Exception:
        app.close()
        raise
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{display_host}:{server.server_port}"
    print(f"数据浏览器：{url}\n数据集：{model.loader.path}\n按 Ctrl+C 停止服务。", flush=True)
    timer = None
    if open_browser:
        timer = Timer(0.5, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止服务…", flush=True)
    finally:
        if timer:
            timer.cancel()
        server.server_close()
        app.close()
