import socket
import struct
import cv2
import numpy as np
import os
import inspect  # <-- add this

from dotenv import load_dotenv
load_dotenv()

from ultralytics import YOLO


class StreamListener:
    """
    TCP client for Pi streaming server.
    Protocol:
      - send 'stream_request\\n'
      - server replies with a single line (e.g., 'OK STREAMING\\n')
      - then a loop of: [4-byte big-endian length][JPEG bytes]
      - to stop: send 'STOP\\n' and close

    Callback signatures supported:
      (A) on_result(result, annotated_frame, raw_frame)
      (B) on_result(result, annotated_frame)   # legacy
          - when there are no detections, annotated_frame will be the raw frame (not None)
    """
    def __init__(self, weights):
        self.HOST = os.getenv("RPI_HOST")
        self.PORT = os.getenv("STREAM_PORT")
        self.REQ_STREAM = bytes(os.getenv("REQ_STREAM") + "\n", "utf-8")
        self.STOP_STREAM = bytes(os.getenv("STOP_STREAM") + "\n", "utf-8")
        self.PING_STREAM = bytes(os.getenv("PING_STREAM") + "\n", "utf-8")

        self.model = YOLO(weights)
        self.sock = None

    # --- low-level helpers ---
    def _connect(self):
        self.sock = socket.create_connection((self.HOST, self.PORT), timeout=5)
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        self.sock.settimeout(None)

    def _readline(self, maxlen=256):
        buf = bytearray()
        while len(buf) < maxlen:
            ch = self.sock.recv(1)
            if not ch:
                return None
            buf += ch
            if ch == b"\n":
                break
        return bytes(buf)

    def _recv_exact(self, n):
        data = bytearray()
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    # --- NEW: smart callback invoker (3-arg or 2-arg) ---
    def _invoke_on_result(self, cb, res, annotated, raw):
        if cb is None:
            return
        try:
            sig = inspect.signature(cb)
            params = len(sig.parameters)
        except Exception:
            # Fallback: try 3-arg, else 2-arg
            params = 3
        try:
            if params >= 3:
                cb(res, annotated, raw)
            else:
                # legacy 2-arg: if no detections, send the raw frame instead of None
                cb(res, annotated if annotated is not None else raw)
        except Exception as e:
            print(f"[StreamListener] on_result error: {e}")

    def req_stream(self):
        if self.sock is None:
            self._connect()
        self.sock.sendall(self.REQ_STREAM)
        header = self._readline()
        print(header.decode("utf-8") if header else "NO HEADER")

    def start_stream_read(self, on_result, on_disconnect, conf_threshold=0.7, show_video=True):
        """
        Connects, requests stream, and processes frames with YOLO.
        Calls:
          - on_result(result, annotated_frame, raw_frame)  # preferred
          - or on_result(result, annotated_frame)          # legacy; raw is used if annotated is None
          - on_disconnect() when the stream ends
        Press ESC to stop (if show_video=True).
        """
        try:
            self.req_stream()

            while True:
                # 1) read 4-byte length header
                hdr = self._recv_exact(4)
                if hdr is None:
                    break
                size = struct.unpack("!I", hdr)[0]
                if size <= 0 or size > 50_000_000:  # sanity check (50MB cap)
                    break

                # 2) read JPEG payload
                jpg = self._recv_exact(size)
                if jpg is None:
                    break

                # 3) decode frame
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    # corrupted frame; skip but still keep UI responsive
                    self._invoke_on_result(on_result, None, None, None)
                    continue

                # 4) YOLO inference
                res = self.model.predict(
                    frame, save=False, imgsz=frame.shape[1],
                    conf=conf_threshold, verbose=False
                )[0]

                # 5) callbacks + optional display
                if len(res.boxes) > 0:
                    annotated = res.plot()
                    self._invoke_on_result(on_result, res, annotated, frame)
                    disp = annotated
                else:
                    # pass raw frame to callback so recorders can use it
                    self._invoke_on_result(on_result, None, None, frame)
                    disp = frame

                if show_video:
                    cv2.imshow("Stream", disp)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        try:
                            self.sock.sendall(self.STOP_STREAM)
                        except Exception:
                            pass
                        break

        finally:
            # ensure cleanup
            try:
                if self.sock:
                    self.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                if self.sock:
                    self.sock.close()
            except Exception:
                pass
            self.sock = None
            if show_video:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            if on_disconnect:
                on_disconnect()

    def close(self):
        try:
            if self.sock:
                try:
                    self.sock.sendall(self.STOP_STREAM)
                except Exception:
                    pass
                self.sock.close()
        finally:
            self.sock = None
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass