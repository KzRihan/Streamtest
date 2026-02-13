#!/usr/bin/env python3
import argparse
import json
import os
import socket
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as exc:
    raise SystemExit("Missing dependency: ultralytics. Install it before running.") from exc

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def _strip_quotes(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == "\"") or (v[0] == v[-1] == "'")):
        return v[1:-1]
    return v


def env_str(*keys: str) -> Optional[str]:
    for key in keys:
        val = os.getenv(key)
        if val:
            return _strip_quotes(val)
    return None


def env_int(*keys: str) -> Optional[int]:
    val = env_str(*keys)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def maybe_load_dotenv() -> None:
    if load_dotenv is None:
        return
    env_file = None
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--env-file" and i + 1 < len(argv):
            env_file = argv[i + 1]
            break
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()


@dataclass
class Detection:
    class_id: int
    label: str
    conf: float
    bbox: Tuple[int, int, int, int]
    ts: float


class RollingDetections:
    def __init__(self, window_s: float) -> None:
        self.window_s = float(window_s)
        self._detections: Deque[Detection] = deque()
        self._best_by_class: Dict[int, Detection] = {}
        self._lock = threading.Lock()

    def update(self, detections: List[Detection], now: Optional[float] = None) -> None:
        now = now if now is not None else time.monotonic()
        with self._lock:
            if detections:
                self._detections.extend(detections)
            removed = self._prune_locked(now)
            if detections or removed:
                self._recompute_locked()

    def snapshot(self) -> Tuple[Optional[Detection], Dict[int, Detection], int, float]:
        now = time.monotonic()
        with self._lock:
            removed = self._prune_locked(now)
            if removed:
                self._recompute_locked()
            best_by_class = dict(self._best_by_class)
            total = len(self._detections)
        best_overall = None
        for det in best_by_class.values():
            if best_overall is None or det.conf > best_overall.conf:
                best_overall = det
        return best_overall, best_by_class, total, now

    def _prune_locked(self, now: float) -> bool:
        cutoff = now - self.window_s
        removed = False
        while self._detections and self._detections[0].ts < cutoff:
            self._detections.popleft()
            removed = True
        return removed

    def _recompute_locked(self) -> None:
        best: Dict[int, Detection] = {}
        for det in self._detections:
            cur = best.get(det.class_id)
            if cur is None or det.conf > cur.conf:
                best[det.class_id] = det
        self._best_by_class = best


def to_numpy(arr) -> np.ndarray:
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        return arr.numpy()
    return np.array(arr)


def load_class_names(path: Optional[str], model_names) -> Dict[int, str]:
    if path:
        names: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    names.append(line)
        return {i: name for i, name in enumerate(names)}
    if isinstance(model_names, dict):
        return {int(k): v for k, v in model_names.items()}
    if isinstance(model_names, (list, tuple)):
        return {i: v for i, v in enumerate(model_names)}
    return {}


def parse_detections(result, label_map: Dict[int, str], now: float) -> List[Detection]:
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes
    xyxy = to_numpy(boxes.xyxy)
    confs = to_numpy(boxes.conf)
    cls_ids = to_numpy(boxes.cls).astype(int)
    detections: List[Detection] = []
    for cls_id, conf, box in zip(cls_ids, confs, xyxy):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        label = label_map.get(int(cls_id), str(int(cls_id)))
        detections.append(Detection(int(cls_id), label, float(conf), (x1, y1, x2, y2), now))
    return detections


class StreamClient:
    def __init__(
        self,
        host: str,
        port: int,
        req_cmd: str,
        stop_cmd: str,
        max_frame_bytes: int = 50_000_000,
    ) -> None:
        self.host = host
        self.port = port
        self.req_cmd = req_cmd
        self.stop_cmd = stop_cmd
        self.max_frame_bytes = max_frame_bytes
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=5)
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        self.sock.settimeout(None)
        self._send_line(self.req_cmd)
        header = self._readline()
        if header:
            print(header.decode("utf-8", "ignore").strip())

    def close(self) -> None:
        if self.sock is None:
            return
        try:
            self._send_line(self.stop_cmd)
        except Exception:
            pass
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass
        self.sock = None

    def _send_line(self, text: Optional[str]) -> None:
        if not text:
            return
        data = (text + "\n").encode("utf-8")
        if self.sock:
            self.sock.sendall(data)

    def _readline(self, maxlen: int = 256) -> Optional[bytes]:
        if self.sock is None:
            return None
        buf = bytearray()
        while len(buf) < maxlen:
            ch = self.sock.recv(1)
            if not ch:
                return None
            buf += ch
            if ch == b"\n":
                break
        return bytes(buf)

    def _recv_exact(self, n: int) -> Optional[bytes]:
        if self.sock is None:
            return None
        data = bytearray()
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    def frames(self):
        while True:
            hdr = self._recv_exact(4)
            if hdr is None:
                break
            size = struct.unpack("!I", hdr)[0]
            if size <= 0 or size > self.max_frame_bytes:
                break
            jpg = self._recv_exact(size)
            if jpg is None:
                break
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            yield frame


class QueryServer(threading.Thread):
    def __init__(self, host: str, port: int, state: RollingDetections) -> None:
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.state = state
        self._stop_event = threading.Event()

    def run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                srv.bind((self.host, self.port))
            except OSError as exc:
                print(f"[query] failed to bind {self.host}:{self.port} ({exc})")
                return
            srv.listen(5)
            srv.settimeout(1.0)
            print(f"[query] listening on {self.host}:{self.port}")
            while not self._stop_event.is_set():
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                self._handle_conn(conn)

    def stop(self) -> None:
        self._stop_event.set()

    def _handle_conn(self, conn: socket.socket) -> None:
        with conn:
            try:
                conn.settimeout(2.0)
                try:
                    conn.recv(256)
                except socket.timeout:
                    pass
                payload = self._build_response()
                conn.sendall(payload.encode("utf-8") + b"\n")
            except Exception:
                pass

    def _build_response(self) -> str:
        best, best_by_class, total, now = self.state.snapshot()

        def det_to_dict(det: Detection) -> Dict:
            return {
                "image_id": det.label,
                "class_id": det.class_id,
                "confidence": round(det.conf, 4),
                "bbox": [int(v) for v in det.bbox],
                "ts": round(det.ts, 3),
            }

        response = {
            "image_id": best.label if best else "NA",
            "class_id": best.class_id if best else -1,
            "confidence": round(best.conf, 4) if best else 0.0,
            "bbox": [int(v) for v in best.bbox] if best else None,
            "window_seconds": self.state.window_s,
            "num_detections": total,
            "per_class": {str(k): det_to_dict(v) for k, v in best_by_class.items()},
            "ts": round(now, 3),
        }
        return json.dumps(response)


def parse_source(value: str) -> Union[int, str]:
    if value.isdigit():
        return int(value)
    return value


def open_video_capture(source: Union[int, str]) -> cv2.VideoCapture:
    if isinstance(source, int):
        if os.name == "nt":
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def iter_video_frames(source: Union[int, str]):
    cap = open_video_capture(source)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Laptop YOLO inference + stream query server")
    parser.add_argument("--env-file", default=None, help="Optional .env file to load")

    weights_default = env_str("YOLO_WEIGHTS", "WEIGHTS")
    parser.add_argument("--weights", default=weights_default, help="Path to YOLO weights")
    parser.add_argument("--classes", default=None, help="Optional classes.txt (one label per line)")
    parser.add_argument("--conf", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=0, help="Inference image size (0 = frame width)")
    parser.add_argument("--infer-fps", type=float, default=0.0, help="Inference FPS limit (0 = every frame)")
    parser.add_argument("--buffer-seconds", type=float, default=1.0, help="Rolling buffer window (seconds)")

    parser.add_argument("--mode", choices=["stream", "self-test"], default="stream")
    parser.add_argument("--source", default="0", help="Self-test source: webcam index or video path")

    parser.add_argument("--stream-host", default=env_str("RPI_HOST"), help="RPi stream host")
    parser.add_argument("--stream-port", type=int, default=env_int("STREAM_PORT"), help="RPi stream port")
    parser.add_argument("--req-cmd", default=env_str("REQ_STREAM") or "REQ", help="Stream request command")
    parser.add_argument("--stop-cmd", default=env_str("STOP_STREAM") or "STOP", help="Stream stop command")

    parser.add_argument("--query-host", default=env_str("QUERY_HOST") or "0.0.0.0", help="Query server host")
    parser.add_argument("--query-port", type=int, default=env_int("QUERY_PORT") or 5052, help="Query server port")

    parser.add_argument("--no-gui", action="store_true", help="Disable display window")
    parser.add_argument("--window-title", default="MDP YOLO Stream", help="OpenCV window title")
    parser.add_argument("--show-fps", action="store_true", help="Overlay FPS on the display")

    args = parser.parse_args()

    if not args.weights:
        parser.error("--weights is required (or set YOLO_WEIGHTS)")
    if args.mode == "stream":
        if not args.stream_host or not args.stream_port:
            parser.error("--stream-host and --stream-port are required for stream mode")
    return args


def main() -> int:
    maybe_load_dotenv()
    args = parse_args()

    model = YOLO(args.weights)
    label_map = load_class_names(args.classes, model.names)
    if args.classes:
        model.names = label_map

    buffer = RollingDetections(args.buffer_seconds)
    query_server = QueryServer(args.query_host, args.query_port, buffer)
    query_server.start()

    last_detections: List[Detection] = []
    last_annotated: Optional[np.ndarray] = None
    last_infer_time = 0.0
    last_frame_time = 0.0
    infer_period = 0.0 if args.infer_fps <= 0 else 1.0 / args.infer_fps

    client: Optional[StreamClient] = None

    try:
        if args.mode == "stream":
            client = StreamClient(
                args.stream_host,
                args.stream_port,
                args.req_cmd,
                args.stop_cmd,
            )
            client.connect()
            frame_iter = client.frames()
        else:
            source = parse_source(args.source)
            frame_iter = iter_video_frames(source)

        for frame in frame_iter:
            now = time.monotonic()
            should_infer = infer_period == 0.0 or (now - last_infer_time) >= infer_period
            if should_infer:
                imgsz = args.imgsz if args.imgsz > 0 else frame.shape[1]
                result = model.predict(
                    frame,
                    imgsz=imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    verbose=False,
                )[0]
                last_detections = parse_detections(result, label_map, now)
                buffer.update(last_detections, now)
                last_annotated = result.plot()
                last_infer_time = now

            annotated = last_annotated if last_annotated is not None else frame

            if args.show_fps:
                annotated = annotated.copy()
                fps = 1.0 / (now - last_frame_time) if last_frame_time else 0.0
                cv2.putText(
                    annotated,
                    f"fps {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            last_frame_time = now

            if not args.no_gui:
                cv2.imshow(args.window_title, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        if client:
            client.close()
        query_server.stop()
        if not args.no_gui:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
