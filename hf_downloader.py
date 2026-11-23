# -*- coding: utf-8 -*-
"""
HF 模型下载器 · Gradio 版
==================================================================
无需代理即可下载 Hugging Face 仓库 / 单文件（走 hf-mirror.com 国内镜像），
也支持一键切回官方原站、wget / curl / aria2c / huggingface_hub / Python 标准库五种下载方式。

运行：
    pip install -r requirements.txt
    python HF模型下载器.py
    # 浏览器打开 http://127.0.0.1:7860

说明：
    界面里的「保存路径」是**运行本程序的那台机器**上的路径（服务端路径）。
"""
from __future__ import annotations

import functools
import inspect
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

try:
    import gradio as gr
except ImportError:
    print("缺少依赖，请先执行： pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

# ------------------------------------------------------------------ 常量
MIRROR = "https://hf-mirror.com"
OFFICIAL = "https://huggingface.co"
DEFAULT_SAVE = str(Path.cwd() / "downloads")
MAX_LOG_LINES = 400
N_OUT = 6  # 所有事件统一更新 6 个输出组件

class UserError(Exception):
    """输入 / 配置类错误，提示语可直接展示给用户。"""

class DownloadError(Exception):
    """下载环节的可恢复错误。"""

class DownloadInterrupted(Exception):
    """用户主动停止。"""

@dataclass
class Entry:
    path: str
    size: int = 0
    lfs: bool = False

@dataclass
class DirEntry:
    path: str
    count: int = 0
    size: int = 0

# ------------------------------------------------------------------ 全局状态（单实例，带锁）
STATE: dict[str, Any] = {
    "parsed": None,
    "files": [],          # list[Entry]
    "dirs": [],           # list[DirEntry]
    "file_map": {},       # path -> Entry
    "dir_map": {},        # path -> DirEntry
    "all_values": [],     # 可勾选的全部 value
    "choices": [],        # (label, value)
    "endpoint": MIRROR,
}
STATE_LOCK = threading.Lock()
LOG_LINES: list[str] = []
PROGRESS = {"text": ""}
LOG_LOCK = threading.Lock()
STOP_EVENT = threading.Event()
DOWNLOAD_LOCK = threading.Lock()

# ------------------------------------------------------------------ 日志 / 小工具
def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")

def log(msg: str) -> None:
    with LOG_LOCK:
        LOG_LINES.append(f"[{_now()}] {msg}")
        if len(LOG_LINES) > MAX_LOG_LINES * 2:
            del LOG_LINES[:-MAX_LOG_LINES]

def set_progress(text: str) -> None:
    with LOG_LOCK:
        PROGRESS["text"] = text

def clear_progress() -> None:
    with LOG_LOCK:
        PROGRESS["text"] = ""

def render_log() -> str:
    with LOG_LOCK:
        lines = LOG_LINES[-MAX_LOG_LINES:]
        if PROGRESS["text"]:
            lines = lines + [PROGRESS["text"]]
    return "\n".join(lines)

def clear_log() -> None:
    with LOG_LOCK:
        LOG_LINES.clear()
        PROGRESS["text"] = ""

def fmt_size(n: float | int | None) -> str:
    if n is None:
        return "—"
    try:
        n = float(n or 0)
    except Exception:
        return "—"
    if n <= 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"

def badge(kind: str, text: str) -> str:
    palette = {
        "idle": ("#6366f1", "rgba(99,102,241,.14)"),
        "ok":   ("#34d399", "rgba(52,211,153,.14)"),
        "run":  ("#38bdf8", "rgba(56,189,248,.14)"),
        "warn": ("#fbbf24", "rgba(251,191,36,.14)"),
        "err":  ("#fb7185", "rgba(251,113,133,.14)"),
    }
    c, bg = palette.get(kind, palette["idle"])
    return (f'<div style="display:inline-flex;align-items:center;gap:8px;padding:7px 14px;'
            f'border-radius:999px;background:{bg};color:{c};border:1px solid {c}55;'
            f'font-size:13px;font-weight:600;white-space:nowrap">{text}</div>')

def friendly_error(e: Exception) -> str:
    """把任何异常翻译成人话 + 可操作建议。"""
    if isinstance(e, UserError):
        return str(e)
    if isinstance(e, DownloadInterrupted):
        return "下载已被手动停止（已完成的文件会保留，可重新执行继续断点续传）"
    s = f"{e}"
    low = (s + " " + type(e).__name__).lower()
    if "no space left" in low or "disk full" in low:
        return "磁盘空间不足，请清理空间或更换保存路径"
    if "permission denied" in low or "not permitted" in low or isinstance(e, PermissionError):
        return "保存路径没有写入权限，请换一个目录或以有权限的账号运行"
    if "401" in s or "unauthorized" in low or "invalid username or password" in low:
        return "认证失败（401）：该仓库可能是 gated / 私有仓库，请填写有效 HF Token，并先在官网申请访问权限"
    if "403" in s or "forbidden" in low or "gated" in low:
        return "访问被拒绝（403）：gated 仓库需登录并接受协议，请填写 HF Token 或改用公开仓库"
    if "404" in s or "not found" in low or "entrynotfound" in low or "repositorynotfound" in low:
        return "未找到（404）：仓库 / 分支 / 文件可能不存在；使用国内镜像时也可能是尚未同步，可切换官方源或稍后重试"
    if "429" in s or "too many requests" in low or "rate limit" in low:
        return "请求过于频繁（429），请稍后重试；aria2c 用户可适当调低线程数"
    if any(k in low for k in ("connection", "timeout", "timed out", "ssl", "max retries", "network", "resolve", "unreachable", "refused")):
        return "网络连接失败：无代理时请选择「国内镜像」；若选官方原站请确认代理可用；也可稍后重试（支持断点续传）"
    if "not installed" in low or "no such file" in low:
        return s
    return f"操作失败：{type(e).__name__}: {s[:300]}"

# ------------------------------------------------------------------ 统一输出打包
def pack(info=None, table=None, sel=None, log_text=None, status=None, single_url=None):
    """与 outputs=[info, table, sel, log, status, single_url] 一一对应。"""
    return (
        info if info is not None else gr.update(),
        table if table is not None else gr.update(),
        sel if sel is not None else gr.update(),
        log_text if log_text is not None else render_log(),
        status if status is not None else gr.update(),
        single_url if single_url is not None else gr.update(),
    )

def error_pack(msg: str):
    return pack(log_text=render_log(), status=badge("err", "✖ " + msg[:70]))

def emit(status=None, **kw):
    """生成一次流式刷新用的元组。"""
    return pack(log_text=render_log(), status=status, **kw)

def safe_gen(fn):
    """所有事件处理器的统一异常兜底：绝不把异常抛给浏览器弹窗。"""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            yield from fn(*args, **kwargs)
        except Exception as e:                       # noqa: BLE001
            msg = friendly_error(e)
            log("❌ " + msg)
            if not isinstance(e, (UserError, DownloadInterrupted)):
                detail = traceback.format_exc(limit=4).strip().replace("\n", " | ")
                log("   排查信息：" + detail[:500])
            clear_progress()
            yield error_pack(msg)
    return wrapper

# ------------------------------------------------------------------ URL / 端点解析
KNOWN_HOSTS = {"huggingface.co", "www.huggingface.co", "hf-mirror.com", "www.hf-mirror.com"}

def current_endpoint(src: str, custom: str) -> str:
    if src == "mirror":
        return MIRROR
    if src == "official":
        return OFFICIAL
    if src == "custom":
        c = (custom or "").strip().strip("/")
        if not c:
            raise UserError("选择了「自定义端点」，请在右侧填写镜像地址，例如 https://hf-mirror.com")
        if not c.startswith(("http://", "https://")):
            c = "https://" + c
        try:
            if not urllib.parse.urlparse(c).netloc:
                raise ValueError
        except Exception:
            raise UserError(f"自定义端点格式不合法：{c}")
        return c.rstrip("/")
    return MIRROR

def apply_endpoint(endpoint: str) -> None:
    """同时覆盖环境变量与模块常量，兼容新旧版 huggingface_hub。"""
    endpoint = endpoint.rstrip("/")
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    try:
        import huggingface_hub.constants as C
        C.ENDPOINT = endpoint
    except Exception:
        pass
    with STATE_LOCK:
        STATE["endpoint"] = endpoint

def parse_hf_url(raw: str) -> dict:
    """支持 repo 简写 / tree / blob / resolve / 子目录 / 分支 / datasets / spaces。"""
    s = (raw or "").strip().strip('"').strip("'")
    if not s:
        raise UserError("请先粘贴 Hugging Face 仓库地址或文件地址")
    s = s.split("#")[0].strip()
    candidate = s if "://" in s else "https://huggingface.co/" + s.lstrip("/")
    try:
        p = urllib.parse.urlparse(candidate)
        if p.scheme.lower() not in ("http", "https") or not p.netloc:
            raise UserError("仅支持 HTTP / HTTPS 地址")
        parts = [x for x in urllib.parse.unquote(p.path).split("/") if x]
        host = p.netloc.lower()
    except Exception as e:
        raise UserError(f"地址解析失败：{e}")

    # 第三方 / CDN 直链：不做仓库结构解析，交给「单文件下载」页
    known = set(KNOWN_HOSTS)
    try:
        known.add(urllib.parse.urlparse(custom_endpoint_hint()).netloc.lower())
    except Exception:
        pass
    if "://" in s and host and host not in known:
        return {"repo_id": "direct", "repo_type": "model", "revision": "main",
                "kind": "file", "sub": "", "host": host, "raw": raw, "is_direct": True}

    repo_type = "model"
    if parts and parts[0] in ("datasets", "spaces"):
        repo_type = {"datasets": "dataset", "spaces": "space"}[parts[0]]
        parts = parts[1:]
    if len(parts) < 2:
        raise UserError("地址里缺少「作者/仓库名」，例如 cmeka/SeedVR2-GGUF")
    repo_id = f"{parts[0]}/{parts[1]}"
    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", repo_id):
        raise UserError(f"仓库名不合法：{repo_id}")

    rest = parts[2:]
    kind, revision, sub = "repo", "main", ""
    if rest:
        head = rest[0]
        if head in ("tree", "blob", "resolve"):
            kind = "tree" if head == "tree" else "file"
            revision = rest[1] if len(rest) > 1 else "main"
            sub = "/".join(rest[2:])
        elif head in ("commits", "commit"):
            revision = rest[1] if len(rest) > 1 else "main"
        else:
            raise UserError(f"无法识别的地址段 /{head}/，支持 tree / blob / resolve / 仓库根地址")

    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision or "main",
        "kind": kind,
        "sub": sub,
        "host": host,
        "raw": raw,
        "is_direct": False,
    }

def custom_endpoint_hint() -> str:
    with STATE_LOCK:
        return str(STATE.get("endpoint") or MIRROR)

def resolve_url(endpoint: str, parsed: dict, rel_path: str) -> str:
    prefix = {"model": "", "dataset": "datasets/", "space": "spaces/"}.get(parsed["repo_type"], "")
    return (f"{endpoint}/{prefix}{parsed['repo_id']}/resolve/"
            f"{urllib.parse.quote(parsed['revision'], safe='')}/"
            f"{urllib.parse.quote(rel_path, safe='/')}?download=true")

# ------------------------------------------------------------------ 仓库文件树
def make_api(endpoint: str, token: str):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise UserError("缺少 huggingface_hub，请执行： pip install -U huggingface_hub")
    kw: dict[str, Any] = {"endpoint": endpoint}
    if token:
        try:
            params = inspect.signature(HfApi.__init__).parameters
            kw["token" if "token" in params else "use_auth_token"] = token
        except Exception:
            kw["token"] = token
    return HfApi(**kw)

def fetch_tree(parsed: dict, endpoint: str, token: str):
    """三级兼容：list_repo_tree -> repo_info(files_metadata) -> list_repo_files。"""
    api = make_api(endpoint, token)
    common = dict(repo_id=parsed["repo_id"], repo_type=parsed["repo_type"], revision=parsed["revision"])
    files: list[Entry] = []
    extra_dirs: set[str] = set()
    errs: list[str] = []

    try:
        for item in api.list_repo_tree(recursive=True, **common):
            name = type(item).__name__
            if name == "RepoFolder":
                extra_dirs.add(getattr(item, "path", ""))
            else:
                files.append(Entry(path=getattr(item, "path", ""),
                                   size=int(getattr(item, "size", 0) or 0),
                                   lfs=bool(getattr(item, "lfs", False))))
    except Exception as e1:
        errs.append(f"list_repo_tree: {type(e1).__name__}")
        try:
            info = api.repo_info(files_metadata=True, **common)
            for sib in (getattr(info, "siblings", None) or []):
                files.append(Entry(path=sib.rfilename,
                                   size=int(getattr(sib, "size", 0) or 0),
                                   lfs=bool(getattr(sib, "lfs", False))))
        except Exception:
            try:
                names = api.list_repo_files(**common)
                files = [Entry(path=n) for n in names]
            except Exception as e3:
                raise DownloadError(f"{friendly_error(e3)}（已尝试 3 种列举接口）")

    files = [f for f in files if f.path]
    if not files:
        raise DownloadError("仓库里没有可下载的文件，或该分支为空")

    # 部分镜像 / 旧接口的 tree 不返回 size，再取一次元数据补齐
    if all(f.size == 0 for f in files):
        try:
            info = api.repo_info(files_metadata=True, **common)
            meta = {s.rfilename: s for s in (getattr(info, "siblings", None) or [])}
            for f in files:
                sib = meta.get(f.path)
                if sib is not None:
                    f.size = int(getattr(sib, "size", 0) or 0)
                    f.lfs = bool(getattr(sib, "lfs", False))
            log("ℹ️ 已通过仓库元数据补齐文件大小信息")
        except Exception:
            pass

    dir_map: dict[str, DirEntry] = {}
    for f in files:
        segs = f.path.split("/")
        for i in range(1, len(segs)):
            d = "/".join(segs[:i])
            de = dir_map.setdefault(d, DirEntry(path=d))
            de.count += 1
            de.size += f.size
    for d in extra_dirs:
        if d:
            dir_map.setdefault(d, DirEntry(path=d))
    dirs = sorted(dir_map.values(), key=lambda x: x.path)
    files.sort(key=lambda x: x.path)
    if errs:
        log("ℹ️ 已使用兼容接口列举文件：" + ", ".join(errs))
    return files, dirs

# ------------------------------------------------------------------ 路径
def build_dest(save_root: str, use_subdir: bool, parsed: dict, rel_path: str):
    raw = (save_root or "").strip().strip('"').strip("'") or DEFAULT_SAVE
    try:
        base = Path(raw).expanduser().resolve()
    except Exception as e:
        raise UserError(f"保存路径不合法：{raw}（{e}）")
    if base.exists() and not base.is_dir():
        raise UserError(f"保存路径是一个文件而不是目录：{base}")
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise UserError(f"无法创建保存目录 {base}：{e}")
    if not os.access(base, os.W_OK):
        raise UserError(f"保存目录不可写：{base}")

    root = base
    if use_subdir:
        root = root / parsed["repo_id"].replace("/", "__")
        if parsed["revision"] not in ("", "main"):
            root = root / parsed["revision"].replace("/", "_")
    root.mkdir(parents=True, exist_ok=True)

    dest = (root / rel_path).resolve()
    try:
        dest.relative_to(root.resolve())
    except ValueError:
        raise UserError("文件路径异常，已拒绝写入到保存目录之外")
    dest.parent.mkdir(parents=True, exist_ok=True)
    return root, dest

def build_direct_dest(save_root: str, single_save: str, filename: str) -> Path:
    s = (single_save or "").strip().strip('"').strip("'")
    filename = Path(urllib.parse.unquote(filename or "download.bin")).name or "download.bin"
    if s:
        try:
            p = Path(s).expanduser().resolve()
        except Exception as e:
            raise UserError(f"另存为路径不合法：{s}（{e}）")
        # 带后缀且不以分隔符结尾 => 视为完整文件路径
        dest = p if (p.suffix and not s.endswith(("/", "\\"))) else (p / filename)
    else:
        base = Path((save_root or "").strip() or DEFAULT_SAVE).expanduser().resolve()
        dest = base / filename
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise UserError(f"无法创建目录 {dest.parent}：{e}")
    if not os.access(dest.parent, os.W_OK):
        raise UserError(f"目标目录不可写：{dest.parent}")
    return dest

def probe_size(root: Path, dest: Path) -> int:
    """优先查看当前下载文件及 Python 临时文件，再查看 Hub incomplete 缓存。"""
    for candidate in (dest, dest.with_name(dest.name + ".part")):
        try:
            if candidate.exists():
                return candidate.stat().st_size
        except OSError:
            pass
    try:
        cache = root / ".cache" / "huggingface" / "download"
        if cache.exists():
            inc = [p for p in cache.rglob("*.incomplete") if p.is_file()]
            if inc:
                return max(p.stat().st_size for p in inc)
    except OSError:
        pass
    return 0


# ------------------------------------------------------------------ 下载执行器
def check_tool(name: str) -> None:
    if not shutil.which(name):
        tips = {
            "wget": "Ubuntu/Debian: sudo apt install wget ｜ macOS: brew install wget ｜ Windows: 放入 PATH",
            "curl": "Ubuntu/Debian: sudo apt install curl ｜ Windows 10+ 自带 curl.exe",
            "aria2c": "Ubuntu/Debian: sudo apt install aria2 ｜ macOS: brew install aria2 ｜ Windows: 下载 aria2 并加入 PATH",
        }
        raise UserError(f"系统未找到 {name}，请先安装（{tips.get(name, '')}），或改用 Python 标准库 / huggingface_hub 方式")

def hub_kwargs(fn, endpoint: str, token: str) -> dict:
    try:
        params = inspect.signature(fn).parameters
    except Exception:
        params = {}
    kw: dict[str, Any] = {}
    if "endpoint" in params:
        kw["endpoint"] = endpoint
    if token:
        kw["token" if "token" in params else "use_auth_token"] = token
    return kw

try:
    from tqdm.auto import tqdm as _TqdmBase
except Exception:                                  # pragma: no cover
    _TqdmBase = None

if _TqdmBase is not None:
    class HubTqdm(_TqdmBase):
        """把 hub 内部的 tqdm 进度映射到界面日志，同时抑制终端刷屏。"""
        def display(self, msg=None, pos=None):
            try:
                total = self.total or 0
                pct = f"（{min(100, self.n * 100 / total):.0f}%）" if total else ""
                set_progress(f"  ⏳ hub 传输中  {fmt_size(self.n)} / {fmt_size(total)}{pct}  {self.desc or ''}")
            except Exception:
                pass
            return True

        def close(self):
            try:
                super().close()
            except Exception:
                pass
            clear_progress()
else:
    HubTqdm = None




class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """跨主机重定向时不把仓库 Token 发送给 CDN / 第三方主机。"""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected:
            old_host = urllib.parse.urlparse(req.full_url).netloc.lower()
            new_host = urllib.parse.urlparse(newurl).netloc.lower()
            if old_host != new_host:
                redirected.remove_header("Authorization")
                redirected.unredirected_hdrs.pop("Authorization", None)
        return redirected

def python_download(url: str, dest: Path, token: str = "", direct_url: bool = False,
                    chunk_size: int = 1024 * 1024, retries: int = 3) -> None:
    """标准库 HTTP 下载器：支持 Range 断点续传、重定向、Bearer Token、重试。"""
    import http.client
    from urllib.request import Request, build_opener
    opener = build_opener(_SafeRedirectHandler())
    part = dest.with_name(dest.name + ".part")
    headers = {"User-Agent": "HF-Model-Downloader/2.0"}
    if token and not direct_url:
        headers["Authorization"] = f"Bearer {token}"
    last_error = None
    for attempt in range(retries + 1):
        try:
            offset = part.stat().st_size if part.exists() else 0
            req_headers = dict(headers)
            if offset:
                req_headers["Range"] = f"bytes={offset}-"
            req = Request(url, headers=req_headers)
            with opener.open(req, timeout=60) as response:
                status = getattr(response, "status", response.getcode())
                if offset and status == 206:
                    cr = response.headers.get("Content-Range", "")
                    if not cr.startswith(f"bytes {offset}-"):
                        raise DownloadError(f"服务器返回了不匹配的 Content-Range：{cr!r}")
                    mode = "ab"
                else:
                    # 服务端不支持 Range（200）时必须从头覆盖，不能把完整响应追加到旧文件。
                    offset = 0
                    mode = "wb"
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(part, mode) as f:
                    while True:
                        block = response.read(chunk_size)
                        if not block:
                            break
                        if STOP_EVENT.is_set():
                            raise DownloadInterrupted()
                        f.write(block)
                    f.flush()
                    os.fsync(f.fileno())
            if not part.exists() or part.stat().st_size <= 0:
                raise DownloadError("服务器响应成功，但下载文件为空")
            os.replace(part, dest)
            return
        except DownloadInterrupted:
            raise
        except Exception as exc:
            # 若服务端报告 416，且 Content-Range 的总长度恰好等于本地偏移，视为已完整下载。
            if getattr(exc, "code", None) == 416 and part.exists():
                try:
                    total = int(exc.headers.get("Content-Range", "").rsplit("/", 1)[1])
                    if part.stat().st_size == total and total > 0:
                        os.replace(part, dest)
                        return
                except Exception:
                    pass
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(2 ** attempt, 8))
    raise DownloadError(f"Python HTTP 下载失败：{type(last_error).__name__}: {last_error}")


def make_worker(method: str, endpoint: str, token: str, parsed: dict,
                rel_path: str, root: Path, dest: Path, aria_parts: int, direct_url: str | None):
    """构造单文件工作线程；外部命令与内置 Python 下载均写入目标路径。"""
    def worker(result: dict) -> None:
        try:
            url = direct_url if direct_url else resolve_url(endpoint, parsed, rel_path)
            if method == "hub" and not direct_url:
                from huggingface_hub import hf_hub_download
                kw = dict(repo_id=parsed["repo_id"], filename=rel_path,
                          revision=parsed["revision"], repo_type=parsed["repo_type"],
                          local_dir=str(root))
                kw.update(hub_kwargs(hf_hub_download, endpoint, token))
                try:
                    params = inspect.signature(hf_hub_download).parameters
                    if "local_dir_use_symlinks" in params:
                        kw["local_dir_use_symlinks"] = False
                    if "tqdm_class" in params and HubTqdm is not None:
                        kw["tqdm_class"] = HubTqdm
                except Exception:
                    pass
                try:
                    result["path"] = hf_hub_download(**kw)
                except (TypeError, AttributeError):
                    kw.pop("tqdm_class", None)
                    result["path"] = hf_hub_download(**kw)
                actual = Path(result["path"])
                if not actual.exists() or actual.stat().st_size <= 0:
                    raise DownloadError("huggingface_hub 返回成功，但本地文件不存在或为空")
                # local_dir 版本通常已写入 dest；路径不一致时复制到统一目标位置。
                if actual.resolve() != dest.resolve():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(actual, dest)
                return

            if method == "python":
                python_download(url, dest, token, direct_url is not None)
                return

            if method == "wget":
                check_tool("wget")
                cmd = ["wget", "--no-verbose", "--tries=3", "--timeout=30",
                       "--continue", "--directory-prefix", str(dest.parent),
                       "--content-disposition", "--trust-server-names"]
                if token and not direct_url:
                    cmd += ["--header", f"Authorization: Bearer {token}"]
                cmd.append(url)
            elif method == "curl":
                check_tool("curl")
                cmd = ["curl", "--fail", "--location", "--retry", "3",
                       "--retry-delay", "2", "--connect-timeout", "30",
                       "--output", str(dest)]
                if token and not direct_url:
                    cmd += ["--header", f"Authorization: Bearer {token}"]
                if dest.exists() and dest.stat().st_size > 0:
                    cmd += ["--continue-at", "-"]
                cmd.append(url)
            elif method == "aria2c":
                check_tool("aria2c")
                cmd = ["aria2c", "--continue=true", "--max-connection-per-server", str(aria_parts),
                       "--split", str(aria_parts), "--min-split-size", "1M",
                       "--console-log-level=warn", "--summary-interval=0",
                       "--dir", str(dest.parent), "--out", dest.name]
                if token and not direct_url:
                    cmd += ["--header", f"Authorization: Bearer {token}"]
                cmd.append(url)
            else:
                raise UserError(f"未知的下载方式：{method}")
            download_started = time.time()
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, errors="replace", cwd=str(dest.parent) if method == "wget" else None)
            result["proc"] = p
            out, _ = p.communicate()
            result["output"] = (out or "")[-2000:]
            result["rc"] = p.returncode
            if p.returncode != 0:
                raise DownloadError(f"{method} 退出码 {p.returncode}：{result['output'][-600:]}")
            if method == "wget" and (not dest.exists() or dest.stat().st_size <= 0):
                # Wget 根据 Content-Disposition / 重定向 URL 命名；统一移动到用户指定文件名。
                candidates = [x for x in dest.parent.iterdir()
                              if x.is_file() and x.name != dest.name
                              and not x.name.endswith((".part", ".aria2"))
                              and x.stat().st_mtime >= download_started - 2]
                if candidates:
                    newest = max(candidates, key=lambda x: x.stat().st_mtime)
                    os.replace(newest, dest)
            if not dest.exists() or dest.stat().st_size <= 0:
                raise DownloadError("命令已结束，但目标文件不存在或为空")
        except Exception as e:
            result["error"] = e
    return worker, ("直链" if direct_url else method)


def run_job(worker, root: Path, dest: Path, label: str, total: int,
            idx: int, count: int, poll_progress: bool = True) -> Iterator[None]:
    """后台执行并轮询；停止时终止子进程，Hub 线程则等待当前请求安全返回。"""
    result: dict[str, Any] = {}
    th = threading.Thread(target=worker, args=(result,), daemon=True)
    t0 = time.time()
    th.start()
    stop_sent = False
    try:
        while th.is_alive():
            if STOP_EVENT.is_set():
                proc = result.get("proc")
                if proc is not None and not stop_sent:
                    try:
                        proc.terminate()
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    stop_sent = True
                set_progress(f"  ⏹ [{idx}/{count}] 正在停止 {label}，等待当前请求退出 …")
            elif poll_progress and not STOP_EVENT.is_set():
                cur = probe_size(root, dest)
                tail = f" / {fmt_size(total)}" if total else ""
                pct = f"（{min(100, cur * 100 / total):.0f}%）" if total else ""
                set_progress(f"  ⏳ [{idx}/{count}] {label}  {fmt_size(cur)}{tail}{pct}")
            yield None
            time.sleep(0.35)
        th.join()
        if STOP_EVENT.is_set():
            raise DownloadInterrupted()
        if result.get("error"):
            raise result["error"]
        dur = max(time.time() - t0, 0.001)
        final = dest.stat().st_size if dest.exists() else 0
        if final <= 0:
            raise DownloadError("下载任务结束，但目标文件不存在或为空")
        clear_progress()
        log(f"  ✅ [{idx}/{count}] {label} · {fmt_size(final)} · {fmt_size(final / dur)}/s · {dur:.1f}s")
        yield None
    finally:
        clear_progress()


def execute_download(items, method, endpoint, token, aria_parts, parsed_override=None):
    """串行下载队列；同一进程只允许一个队列，防止全局停止标志和状态互相覆盖。"""
    if not items:
        raise UserError("下载列表为空")
    if not DOWNLOAD_LOCK.acquire(blocking=False):
        raise UserError("已有下载任务正在运行，请等待完成或先停止当前任务")
    STOP_EVENT.clear()
    total_n = len(items)
    ok, failed, stopped = 0, [], False
    try:
        for i, (label, root, dest, size, direct_url, rel_path) in enumerate(items, 1):
            if STOP_EVENT.is_set():
                stopped = True
                log("⏹ 收到停止指令，剩余任务已跳过")
                break
            parsed = parsed_override
            if direct_url is None:
                if parsed is None:
                    with STATE_LOCK:
                        parsed = STATE.get("parsed")
                if not parsed:
                    raise UserError("请先点击「查询仓库文件」再下载")
                if dest.exists() and size and dest.stat().st_size == size:
                    log(f"⏭ [{i}/{total_n}] 已完整存在，跳过 {label}")
                    ok += 1
                    yield emit(status=badge("run", f"⏬ 下载中 {i}/{total_n}"))
                    continue
            else:
                parsed = {}
            log(f"▶ [{i}/{total_n}] {label} → {dest}")
            yield emit(status=badge("run", f"⏬ 下载中 {i}/{total_n}"))
            worker, _ = make_worker(method, endpoint, token, parsed or {}, rel_path or "",
                                    root, dest, aria_parts, direct_url)
            try:
                for _ in run_job(worker, root, dest, label, size, i, total_n,
                                 poll_progress=(method != "hub")):
                    yield emit(status=badge("run", f"⏬ 下载中 {i}/{total_n}"))
                ok += 1
            except DownloadInterrupted:
                stopped = True
                yield emit(status=badge("warn", "⏹ 已停止"))
                break
            except Exception as e:
                msg = friendly_error(e)
                failed.append((label, msg))
                log(f"  ❌ {label}：{msg}")
                yield emit(status=badge("warn", f"⚠ 第 {i} 个文件失败"))
        clear_progress()
        if stopped:
            log(f"⏹ 下载已停止：完成 {ok} 个，失败 {len(failed)} 个，队列剩余项未执行")
            yield emit(status=badge("warn", f"⏹ 已停止 · 完成 {ok}/{total_n}"))
        elif failed:
            log(f"—— 完成 {ok} 个，失败 {len(failed)} 个（失败原因见上方日志）——")
            yield emit(status=badge("warn", f"⚠ 完成 {ok}/{total_n}，{len(failed)} 个失败"))
        else:
            log(f"🎉 全部完成：{ok} 个文件，已保存到指定目录")
            yield emit(status=badge("ok", f"✔ 下载完成 {ok} 个文件"))
    finally:
        clear_progress()
        DOWNLOAD_LOCK.release()


# ------------------------------------------------------------------ 选择展开
def expand_selection(selected: list[str]) -> list[str]:
    with STATE_LOCK:
        files = set(STATE["file_map"].keys())
        dirs = set(STATE["dir_map"].keys())
    want: set[str] = set()
    for s in selected or []:
        if s in dirs or s not in files:
            prefix = s.rstrip("/") + "/"
            want.update(p for p in files if p.startswith(prefix))
        else:
            want.add(s)
    return sorted(want)

# ------------------------------------------------------------------ Gradio 事件
@safe_gen
def on_query(url, src, custom, token):
    if DOWNLOAD_LOCK.locked():
        raise UserError("当前下载任务尚未结束，请等待完成或停止后再查询新仓库")
    endpoint = current_endpoint(src, custom)
    apply_endpoint(endpoint)
    parsed = parse_hf_url(url)
    if parsed["is_direct"]:
        log("ℹ️ 这是第三方直链，已自动填入「单文件下载」标签页")
        yield pack(single_url=parsed["raw"], log_text=render_log(),
                   status=badge("warn", "👉 请切到「单文件下载」标签页"))
        raise UserError("第三方 / CDN 直链请在「单文件下载」标签页下载（已为你回填地址）")

    log(f"🔍 查询 {parsed['repo_id']} · {parsed['revision']} · {endpoint}")
    yield emit(status=badge("run", "🔍 正在列举仓库文件…"), single_url=parsed["raw"] if parsed["kind"] == "file" else None)
    files, dirs = fetch_tree(parsed, endpoint, token)

    with STATE_LOCK:
        STATE.update({
            "parsed": parsed, "files": files, "dirs": dirs,
            "file_map": {f.path: f for f in files},
            "dir_map": {d.path: d for d in dirs},
        })

    rows = []
    choices = []
    n = 0
    for d in dirs:
        n += 1
        rows.append([n, "📁 " + d.path + "/", "文件夹", f"{d.count} 项 · {fmt_size(d.size)}", "—"])
        choices.append((f"📁 {d.path}/    ·  {d.count} 个文件 · {fmt_size(d.size)}", d.path))
    for f in files:
        n += 1
        rows.append([n, "📄 " + f.path, "文件", fmt_size(f.size), "✔ LFS" if f.lfs else "—"])
        choices.append((f"📄 {f.path}    ·  {fmt_size(f.size)}", f.path))
    with STATE_LOCK:
        STATE["choices"] = choices
        STATE["all_values"] = [v for _, v in choices]

    total = sum(f.size for f in files)
    lfs_n = sum(1 for f in files if f.lfs)
    default_sel = []
    sub = parsed["sub"]
    if sub:
        if sub in STATE["dir_map"] or sub.rstrip("/") in STATE["dir_map"]:
            default_sel = [sub.rstrip("/")]
        elif sub in STATE["file_map"]:
            default_sel = [sub]
        else:
            matched = [p for p in STATE["file_map"] if p.startswith(sub.rstrip("/") + "/")]
            if matched:
                default_sel = matched[:50]
    host = urllib.parse.urlparse(endpoint).netloc
    tip = ("检测到你粘贴的是文件地址，已同步填入「单文件下载」标签页；"
           "下面也列出了整个仓库，可直接勾选下载。" if parsed["kind"] == "file" else "")
    if sub and default_sel:
        tip += f" 已自动选中地址中指向的 `{sub}`。"
    info = f"""
### ✅ 查询成功 · `{parsed['repo_id']}`
| 项目 | 值 |
| --- | --- |
| 下载源 | `{host}`（{('国内镜像' if src == 'mirror' else '官方原站' if src == 'official' else '自定义')}） |
| 仓库类型 / 版本 | `{parsed['repo_type']}` · `{parsed['revision']}` |
| 文件 / 文件夹 | **{len(files)}** 个文件 · **{len(dirs)}** 个文件夹 |
| 合计大小 / LFS | **{fmt_size(total)}** · {lfs_n} 个 LFS 大文件 |

{tip}
> 在下方勾选框里选择 1 个或多个条目；**选中文件夹会递归下载其下所有文件**。
"""
    log(f"✅ 共 {len(files)} 个文件 / {len(dirs)} 个文件夹，合计 {fmt_size(total)}")
    yield pack(info=info, table=rows,
               sel=gr.update(choices=choices, value=default_sel),
               log_text=render_log(), status=badge("ok", f"✔ 已列出 {len(files)} 个文件"),
               single_url=parsed["raw"] if parsed["kind"] == "file" else gr.update())

@safe_gen
def on_select_all(src, custom):
    with STATE_LOCK:
        values = list(STATE["all_values"])
    log(f"☑ 全选 {len(values)} 项")
    yield emit(sel=gr.update(value=values), status=badge("idle", f"☑ 已选 {len(values)} 项"))

@safe_gen
def on_select_files():
    with STATE_LOCK:
        values = list(STATE["file_map"].keys())
    log(f"☑ 仅选文件 {len(values)} 项")
    yield emit(sel=gr.update(value=values), status=badge("idle", f"☑ 已选 {len(values)} 个文件"))

@safe_gen
def on_select_none():
    log("☐ 已清空选择")
    yield emit(sel=gr.update(value=[]), status=badge("idle", "☐ 未选择"))

@safe_gen
def on_invert(selected):
    with STATE_LOCK:
        allv = list(STATE["all_values"])
    val = [v for v in allv if v not in (selected or [])]
    log(f"🔄 反选，当前 {len(val)} 项")
    yield emit(sel=gr.update(value=val), status=badge("idle", f"☑ 已选 {len(val)} 项"))

@safe_gen
def on_filter(keyword, selected):
    with STATE_LOCK:
        choices = list(STATE["choices"])
    k = (keyword or "").strip().lower()
    filtered = [c for c in choices if not k or k in c[0].lower()]
    kept = [v for v in (selected or []) if any(v == val for _, val in filtered)]
    dropped = len(selected or []) - len(kept)
    log(f"🔎 筛选「{keyword or '全部'}」→ {len(filtered)} 项可见" + (f"，{dropped} 个被隐藏的勾选已取消" if dropped else ""))
    yield emit(sel=gr.update(choices=filtered, value=kept),
               status=badge("idle", f"🔎 {len(filtered)} 项可见"))

def _prepare(parsed, repo_url, save_root, use_subdir, src, custom, token, method):
    if not parsed:
        raise UserError("请先点击「🔍 查询仓库文件」再执行下载")
    current = parse_hf_url(repo_url)
    keys = ("repo_id", "repo_type", "revision")
    if any(current.get(k) != parsed.get(k) for k in keys):
        raise UserError("仓库地址、仓库类型或版本与上次查询不一致，请重新点击「查询仓库文件」")
    endpoint = current_endpoint(src, custom)
    apply_endpoint(endpoint)
    if method in ("wget", "curl", "aria2c"):
        check_tool(method)
    return endpoint


@safe_gen
def on_download_sel(repo_url, src, custom, method, save_root, use_subdir, token, aria, selected):
    with STATE_LOCK:
        parsed = dict(STATE["parsed"]) if STATE["parsed"] else None
        fmap = dict(STATE["file_map"])
    endpoint = _prepare(parsed, repo_url, save_root, use_subdir, src, custom, token, method)
    if not (selected or []):
        raise UserError("还没有勾选任何条目，请先查询并在文件清单里勾选（或点「下载整个仓库」）")
    paths = expand_selection(selected)
    if not paths:
        raise UserError("所选条目下没有可下载的文件")
    log(f"⬇ 开始下载选中项：{len(selected)} 个条目 → 展开为 {len(paths)} 个文件 · 方式 {method}")
    yield emit(status=badge("run", f"⏬ 准备下载 {len(paths)} 个文件"))
    items = []
    for p in paths:
        root, dest = build_dest(save_root, use_subdir, parsed, p)
        items.append((p, root, dest, fmap.get(p, Entry(p)).size, None, p))
    yield from execute_download(items, method, endpoint, token, int(aria), parsed_override=parsed)

@safe_gen
def on_download_all(repo_url, src, custom, method, save_root, use_subdir, token, aria):
    with STATE_LOCK:
        parsed = dict(STATE["parsed"]) if STATE["parsed"] else None
        fmap = dict(STATE["file_map"])
    endpoint = _prepare(parsed, repo_url, save_root, use_subdir, src, custom, token, method)
    paths = sorted(fmap.keys())
    if not paths:
        raise UserError("文件清单为空，请先点击「🔍 查询仓库文件」")
    log(f"📦 下载整个仓库：{len(paths)} 个文件 · 方式 {method}")
    yield emit(status=badge("run", f"📦 整仓 {len(paths)} 个文件"))
    items = []
    for p in paths:
        root, dest = build_dest(save_root, use_subdir, parsed, p)
        items.append((p, root, dest, fmap[p].size, None, p))
    yield from execute_download(items, method, endpoint, token, int(aria), parsed_override=parsed)

@safe_gen
def on_download_single(single_url, single_save, src, custom, method,
                       save_root, use_subdir, token, aria):
    raw = (single_url or "").strip()
    if not raw:
        raise UserError("请粘贴单个文件地址，例如 https://huggingface.co/cmeka/SeedVR2-GGUF/blob/main/xxx.gguf")
    endpoint = current_endpoint(src, custom)
    apply_endpoint(endpoint)
    actual_method = method
    direct_url = None
    parsed = None
    rel_path = ""

    parsed_try = parse_hf_url(raw)
    if parsed_try["is_direct"]:
        direct_url = raw
        filename = urllib.parse.unquote(urllib.parse.urlparse(raw).path).rstrip("/").split("/")[-1]
        filename = filename or "download.bin"
        if method == "hub":
            actual_method = "python"
            log("ℹ️ 任意直链不支持 huggingface_hub，已自动改用 Python 标准库 HTTP 下载")
        dest = build_direct_dest(save_root, single_save, filename)
        root = dest.parent
        label = filename
        size = 0
    else:
        if parsed_try["kind"] != "file":
            raise UserError("这看起来是仓库地址而不是文件地址；请切到「仓库下载」标签页，或粘贴 blob / resolve 文件链接")
        if not parsed_try["sub"]:
            raise UserError("文件地址里缺少文件名（应为 /blob/分支/文件名 或 /resolve/分支/文件名）")
        parsed = parsed_try
        rel_path = parsed_try["sub"]
        root, dest = build_dest(save_root, use_subdir, parsed, rel_path)
        if (single_save or "").strip():
            dest = build_direct_dest(save_root, single_save, Path(rel_path).name)
        with STATE_LOCK:
            fmap = dict(STATE["file_map"])
        size = fmap.get(rel_path, Entry(rel_path)).size
        label = rel_path
        if actual_method in ("wget", "curl", "aria2c"):
            check_tool(actual_method)

    log(f"⬇ 单文件下载：{label} · 源 {urllib.parse.urlparse(endpoint).netloc} · 方式 {actual_method}")
    yield emit(status=badge("run", "⏬ 单文件下载中"))
    items = [(label, root, dest, size, direct_url, rel_path)]
    yield from execute_download(items, actual_method, endpoint, token, int(aria),
                                parsed_override=parsed)

@safe_gen
def on_stop():
    STOP_EVENT.set()
    log("⏹ 已请求停止：当前文件（或当前命令）结束后终止，已完成的文件保留")
    yield emit(status=badge("warn", "⏹ 正在停止…"))

@safe_gen
def on_clear_log():
    clear_log()
    yield pack(log_text="", status=badge("idle", "🧹 日志已清空"))

@safe_gen
def on_src_change(src, custom):
    try:
        ep = current_endpoint(src, custom)
        apply_endpoint(ep)
        log(f"🔀 下载源切换为 {ep}")
        yield emit(status=badge("ok", "🔀 " + urllib.parse.urlparse(ep).netloc))
    except Exception as e:
        raise UserError(friendly_error(e))

# ------------------------------------------------------------------ UI
CUSTOM_CSS = """
:root { --gh-line: rgba(148,163,184,.16); }
body, .gradio-container {
  background: radial-gradient(1100px 600px at 12% -10%, rgba(124,58,237,.22), transparent 60%),
              radial-gradient(900px 520px at 92% 0%, rgba(6,182,212,.18), transparent 60%),
              #080d1a !important;
  color: #e6ecf7 !important;
  font-family: "PingFang SC","Microsoft YaHei","WenQuanYi Micro Hei",-apple-system,sans-serif !important;
}
.gradio-container { max-width: 1320px !important; padding: 22px 26px 40px !important; }
.gh-hero {
  position:relative; overflow:hidden; padding: 26px 30px; margin-bottom: 20px;
  border-radius: 22px; border: 1px solid rgba(167,139,250,.28);
  background: linear-gradient(115deg, rgba(99,102,241,.20), rgba(6,182,212,.13) 55%, rgba(236,72,153,.13));
  box-shadow: 0 18px 50px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.08);
}
.gh-hero:after { content:""; position:absolute; right:-70px; top:-90px; width:280px; height:280px;
  background: radial-gradient(circle, rgba(167,139,250,.35), transparent 70%); filter: blur(8px); }
.gh-hero h1 { margin:0; font-size:30px; letter-spacing:.5px; font-weight:800;
  background: linear-gradient(90deg,#fff,#a5b4fc 55%,#67e8f9); -webkit-background-clip:text; color:transparent; }
.gh-hero p { margin:8px 0 0; color:#aebbd4; font-size:13.5px; line-height:1.7; }
.gh-chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
.gh-chips span { font-size:12px; padding:5px 11px; border-radius:999px; color:#c7d2fe;
  background:rgba(99,102,241,.16); border:1px solid rgba(129,140,248,.30); }
.gh-card { border:1px solid var(--gh-line) !important; border-radius:16px !important;
  background: rgba(17,25,45,.62) !important; backdrop-filter: blur(12px);
  box-shadow: 0 10px 30px rgba(0,0,0,.22) !important; padding: 14px 16px !important; }
.gr-box, .gr-panel, .gr-accordion, .gr-tabitem { border-color: var(--gh-line) !important; }
label span, .gr-markdown, .gr-markdown p { color:#d9e2f3 !important; }
input[type=text], input[type=password], textarea {
  background: rgba(9,15,30,.75) !important; color:#e8eefb !important;
  border:1px solid rgba(148,163,184,.25) !important; border-radius:10px !important; }
.gr-radio label, .gr-checkbox label { border-radius:10px !important; }
#selbox { max-height: 330px; overflow-y:auto; padding: 10px 12px;
  border:1px solid var(--gh-line); border-radius:12px; background:rgba(9,15,30,.55); }
#selbox label { display:flex !important; align-items:center; gap:8px; padding:7px 10px; margin:3px 0;
  border-radius:9px; background:rgba(255,255,255,.035); border:1px solid transparent;
  font-size:12.5px; line-height:1.45; transition:.18s; }
#selbox label:hover { background:rgba(129,140,248,.16); border-color:rgba(129,140,248,.35); }
#selbox .grid-wrap, #selbox .gr-checkbox-group { display:grid !important;
  grid-template-columns: repeat(auto-fill, minmax(360px,1fr)); gap:4px 10px; }
#logbox textarea { background:#050912 !important; color:#7ee787 !important;
  font-family: "JetBrains Mono","Cascadia Code",Consolas,"WenQuanYi Micro Hei",monospace !important;
  font-size:12px !important; line-height:1.65 !important; border:1px solid rgba(52,211,153,.18) !important; }
#btn-query button, #btn-dl-sel button, #btn-dl-all button, #btn-dl-one button {
  background: linear-gradient(115deg,#6366f1,#8b5cf6 48%,#06b6d4) !important;
  color:#fff !important; font-weight:700 !important; border:none !important;
  border-radius:11px !important; box-shadow:0 8px 22px rgba(99,102,241,.35) !important; }
#btn-query button:hover, #btn-dl-sel button:hover, #btn-dl-all button:hover, #btn-dl-one button:hover {
  filter:brightness(1.14); transform:translateY(-1px); }
#btn-stop button { border-radius:11px !important; }
.gh-table .table-wrap { max-height: 340px; overflow:auto; border-radius:10px; }
.gh-table table { font-size:12.5px !important; }
.tabs > .tab-nav { gap:6px !important; }
.tabs > .tab-nav button { border-radius:10px 10px 0 0 !important; font-weight:600 !important; }
footer { display:none !important; }
::-webkit-scrollbar { width:9px; height:9px; }
::-webkit-scrollbar-thumb { background:rgba(129,140,248,.35); border-radius:9px; }
::-webkit-scrollbar-track { background:transparent; }
"""

HERO = """
<div class="gh-hero">
  <h1>🛰️ HF 模型下载器</h1>
  <p>无代理也能拉取 Hugging Face 模型 · 国内镜像 / 官方原站一键切换 · 整仓 / 子目录 / 单文件 / 任意直链全支持 · 断点续传</p>
  <div class="gh-chips">
    <span>🌉 hf-mirror.com 免代理</span><span>🐍 huggingface_hub</span>
    <span>📦 wget</span><span>🔗 curl</span><span>⚡ aria2c 多线程</span><span>🐍 Python HTTP</span>
    <span>📁 文件夹递归</span><span>⏯ 断点续传 / 可停止</span>
  </div>
</div>
"""

WELCOME = """
### 👋 使用三步走
1. 在上方选好**下载源**（无代理请保持「国内镜像」）和**保存路径**；
2. 粘贴仓库地址 → 点 **🔍 查询仓库文件** → 在清单里勾选文件 / 文件夹（文件夹会自动递归）；
3. 点 **⬇ 下载选中项** 或 **📦 下载整个仓库**，底部日志会实时显示进度。

> 单文件（blob / resolve 链接、任意 http 直链）请切到 **「单文件下载」** 标签页。
> 保存路径是**运行本程序的机器**上的路径；下载中断后直接重跑同一操作即可续传。
"""

def make_theme():
    try:
        return gr.themes.Soft(primary_hue="violet", secondary_hue="cyan",
                              neutral_hue="slate", text_size="md", spacing_size="md")
    except Exception:
        return gr.themes.Soft(primary_hue="violet")

try:
    GR_MAJOR = int(str(gr.__version__).split(".")[0])
except Exception:
    GR_MAJOR = 4

def build_ui():
    blocks_kw = {"title": "HF 模型下载器 · 免代理下载 Hugging Face",
                 "analytics_enabled": False}
    if GR_MAJOR < 6:                       # 4.x / 5.x 在 Blocks 上传主题
        blocks_kw.update(theme=make_theme(), css=CUSTOM_CSS)
    with gr.Blocks(**blocks_kw) as demo:
        gr.HTML(HERO)

        # ---------- 全局设置 ----------
        with gr.Row(elem_classes=["gh-card"]):
            src = gr.Radio(
                [("🌉 国内镜像 hf-mirror.com（推荐 · 无需代理）", "mirror"),
                 ("🌐 官方原站 huggingface.co（需代理）", "official"),
                 ("⚙️ 自定义端点", "custom")],
                value="mirror", label="① 下载源", scale=2)
            custom_host = gr.Textbox(value="", placeholder="https://your-mirror.example.com",
                                     label="自定义端点（选「自定义」时填写）", scale=2)
            method = gr.Radio(
                [("🐍 huggingface_hub（推荐 · 断点续传 + 缓存）", "hub"),
                 ("📦 wget", "wget"), ("🔗 curl", "curl"), ("⚡ aria2c 多线程", "aria2c"), ("🐍 Python 标准库（无需外部程序）", "python")],
                value="hub", label="② 下载方式", scale=3)
        with gr.Row(elem_classes=["gh-card"]):
            save_root = gr.Textbox(value=DEFAULT_SAVE, label="③ 保存根目录（服务端绝对路径或相对路径）", scale=3)
            use_subdir = gr.Checkbox(value=True, label="按「仓库名 / 版本」自动建子目录", scale=1)
            aria = gr.Slider(1, 16, value=8, step=1, label="aria2c 线程数", scale=1)
            token = gr.Textbox(value="", type="password", label="HF Token（可选 · gated / 私有仓库）", scale=2)
        status_html = gr.HTML(badge("idle", "🟢 就绪 · 当前源 hf-mirror.com"))

        with gr.Tabs():
            # ---------- Tab 1 仓库 ----------
            with gr.Tab("📚 仓库下载（可多选 / 文件夹递归）"):
                repo_url = gr.Textbox(
                    label="仓库地址",
                    placeholder="https://huggingface.co/cmeka/SeedVR2-GGUF/tree/main   （也支持 cmeka/SeedVR2-GGUF 简写）",
                    value="https://huggingface.co/cmeka/SeedVR2-GGUF/tree/main")
                with gr.Row():
                    btn_query = gr.Button("🔍 查询仓库文件", elem_id="btn-query", scale=2)
                    btn_all = gr.Button("📦 下载整个仓库", elem_id="btn-dl-all", scale=2)
                    btn_sel = gr.Button("⬇ 下载选中项", elem_id="btn-dl-sel", scale=2)
                    btn_stop = gr.Button("⏹ 停止", elem_id="btn-stop", scale=1)
                info = gr.Markdown(WELCOME, elem_classes=["gh-card"])
                table = gr.DataFrame(headers=["#", "路径", "类型", "大小", "LFS"],
                                     datatype=["number", "str", "str", "str", "str"],
                                     interactive=False, label="📋 仓库文件清单",
                                     elem_id="filetable", elem_classes=["gh-table"])
                with gr.Row(elem_classes=["gh-card"]):
                    keyword = gr.Textbox(value="", label="🔎 按关键字过滤清单（如 Q4_0 / vae / .safetensors）", scale=3)
                    btn_filter = gr.Button("筛选", scale=1)
                    btn_sel_all = gr.Button("全选", scale=1)
                    btn_sel_files = gr.Button("仅选文件", scale=1)
                    btn_invert = gr.Button("反选", scale=1)
                    btn_sel_none = gr.Button("清空", scale=1)
                sel = gr.CheckboxGroup(choices=[], value=[], label="✅ 选择要下载的文件 / 文件夹（可多选，文件夹递归）",
                                       elem_id="selbox")

            # ---------- Tab 2 单文件 ----------
            with gr.Tab("📄 单文件下载（blob / resolve / 任意直链）"):
                single_url = gr.Textbox(
                    label="单个文件地址",
                    placeholder="https://huggingface.co/cmeka/SeedVR2-GGUF/blob/main/xxx.gguf\n或 https://hf-mirror.com/.../resolve/main/xxx.gguf\n或任意 http(s) 直链")
                single_save = gr.Textbox(value="", label="另存为（可选：可填目录，也可填含扩展名的完整文件路径）",
                                         placeholder="留空则保存到「保存根目录 / 仓库名 / 原相对路径」")
                with gr.Row():
                    btn_one = gr.Button("⬇ 下载此文件", elem_id="btn-dl-one", scale=3)
                    gr.HTML("<div style='font-size:12px;color:#94a3b8;line-height:2.4'>支持 blob / resolve 链接；第三方直链会自动用 Python 标准库下载</div>", scale=4)
                gr.Markdown("""
**地址示例**
- `https://huggingface.co/cmeka/SeedVR2-GGUF/blob/main/model_Q4_0.gguf`
- `https://hf-mirror.com/cmeka/SeedVR2-GGUF/resolve/main/model_Q4_0.gguf?download=true`
- `cmeka/SeedVR2-GGUF/blob/main/vae/config.json`（简写）
- 任意 `https://.../xxx.safetensors` 直链（自动降级为 Python HTTP）
""", elem_classes=["gh-card"])

        # ---------- 日志 ----------
        with gr.Row(elem_classes=["gh-card"]):
            with gr.Column(scale=5):
                gr.Markdown("### 🖥️ 实时日志 / 进度")
                logbox = gr.Textbox(value="", lines=16, max_lines=16, show_label=False,
                                    elem_id="logbox", interactive=False)
            with gr.Column(scale=1, min_width=140):
                gr.Markdown("**操作**")
                btn_clear_log = gr.Button("🧹 清空日志")
                gr.Markdown("**状态**")
                gr.Markdown("""
<div style="font-size:12px;color:#94a3b8;line-height:1.9">
· 中断后<b>重跑即可续传</b><br>
· 404 多为镜像未同步，可切官方源<br>
· 401/403 请填 HF Token<br>
· 大文件建议 aria2c
</div>""")

        outs = [info, table, sel, logbox, status_html, single_url]
        common = [src, custom_host, method, save_root, use_subdir, token, aria]

        # 绑定
        btn_query.click(fn=on_query, inputs=[repo_url, src, custom_host, token], outputs=outs)
        btn_sel.click(fn=on_download_sel,
                      inputs=[repo_url, src, custom_host, method, save_root, use_subdir, token, aria, sel],
                      outputs=outs)
        btn_all.click(fn=on_download_all,
                      inputs=[repo_url, src, custom_host, method, save_root, use_subdir, token, aria],
                      outputs=outs)
        btn_one.click(fn=on_download_single,
                      inputs=[single_url, single_save, src, custom_host, method, save_root, use_subdir, token, aria],
                      outputs=outs)
        btn_filter.click(fn=on_filter, inputs=[keyword, sel], outputs=outs)
        keyword.submit(fn=on_filter, inputs=[keyword, sel], outputs=outs)
        btn_sel_all.click(fn=on_select_all, inputs=[src, custom_host], outputs=outs)
        btn_sel_files.click(fn=on_select_files, inputs=[], outputs=outs)
        btn_invert.click(fn=on_invert, inputs=[sel], outputs=outs)
        btn_sel_none.click(fn=on_select_none, inputs=[], outputs=outs)
        btn_stop.click(fn=on_stop, inputs=[], outputs=outs)
        btn_clear_log.click(fn=on_clear_log, inputs=[], outputs=outs)
        src.change(fn=on_src_change, inputs=[src, custom_host], outputs=outs)

        gr.Markdown(
            "<div style='text-align:center;color:#64748b;font-size:12px;margin-top:14px'>"
            "HF 模型下载器 · 基于 hf-mirror.com 公益镜像 · 请遵守模型许可证与站点使用条款</div>")
    return demo


demo = build_ui()

if __name__ == "__main__":
    port = int(os.environ.get("HF_DL_PORT", "7860"))
    print(f"\n🛰️  HF 模型下载器启动中 → http://127.0.0.1:{port}")
    print(f"    默认下载源：{MIRROR}（无代理可用）  默认保存目录：{DEFAULT_SAVE}\n")
    try:
        launch_kw = {"server_name": "127.0.0.1", "server_port": port,
                     "inbrowser": True, "share": False}
        if GR_MAJOR >= 6:                 # 6.x 起主题/样式在 launch 传入
            launch_kw.update(theme=make_theme(), css=CUSTOM_CSS)
        demo.queue().launch(**launch_kw)
    except Exception as e:                       # noqa: BLE001
        print(f"启动失败：{friendly_error(e)}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
