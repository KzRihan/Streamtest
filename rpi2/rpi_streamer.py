#!/usr/bin/env python3
import argparse
import os
import sys

from stream_server import StreamServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RPi streaming server (JPEG over TCP)")
    parser.add_argument("--host", default=os.getenv("RPI_HOST") or "0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=int(os.getenv("STREAM_PORT", "5001")), help="Bind port")
    parser.add_argument("--buffer-size", type=int, default=int(os.getenv("BUFFER_SIZE", "65536")), help="Socket send buffer")
    parser.add_argument("--req-cmd", default=os.getenv("REQ_STREAM") or "REQ", help="Stream request command")
    parser.add_argument("--stop-cmd", default=os.getenv("STOP_STREAM") or "STOP", help="Stream stop command")
    parser.add_argument("--ping-cmd", default=os.getenv("PING_STREAM") or "PING", help="Stream ping command")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.environ["RPI_HOST"] = str(args.host)
    os.environ["STREAM_PORT"] = str(args.port)
    os.environ["BUFFER_SIZE"] = str(args.buffer_size)
    os.environ["REQ_STREAM"] = str(args.req_cmd)
    os.environ["STOP_STREAM"] = str(args.stop_cmd)
    os.environ["PING_STREAM"] = str(args.ping_cmd)

    srv = StreamServer()
    try:
        srv.connect()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        srv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
