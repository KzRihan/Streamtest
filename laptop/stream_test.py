import argparse
import os
import sys
import time
import cv2

# Adjust this import to match your file name that defines StreamListener.
# e.g., if your class is in stream_client.py, use:
# from stream_client import StreamListener
from StreamListener import StreamListener


def main():
    parser = argparse.ArgumentParser(description="Test StreamListener with YOLO")
    parser.add_argument("--weights", required=True, help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--no-gui", default=False, help="Disable live window")
    parser.add_argument("--save-dir", default="", help="Directory to save annotated frames")
    args = parser.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    listener = StreamListener(weights=args.weights)

    last_t = None
    frame_idx = 0

    def on_result(res, annotated_frame):
        nonlocal last_t, frame_idx
        frame_idx += 1
        now = time.perf_counter()
        fps = 1.0 / (now - last_t) if last_t else 0.0
        last_t = now

        if res is None:
            # No detections
            print(f"[{frame_idx:06d}] no detections | fps={fps:.1f}", end="\r")
        else:
            # Summarize detections
            names = res.names
            cls_ids = res.boxes.cls.tolist() if hasattr(res.boxes.cls, "tolist") else res.boxes.cls
            confs = res.boxes.conf.tolist() if hasattr(res.boxes.conf, "tolist") else res.boxes.conf
            dets = ", ".join(f"{names[int(c)]}:{conf:.2f}" for c, conf in zip(cls_ids, confs))
            print(f"[{frame_idx:06d}] {len(cls_ids)} det(s): {dets} | fps={fps:.1f}   ")

            # Save if requested
            if args.save_dir and annotated_frame is not None:
                out_path = os.path.join(args.save_dir, f"frame_{frame_idx:06d}.jpg")
                try:
                    cv2.imwrite(out_path, annotated_frame)
                except Exception as e:
                    print(f"\nFailed to save {out_path}: {e}")

    def on_disconnect():
        print("\nDisconnected from server.")

    try:
        listener.start_stream_read(
            on_result=on_result,
            on_disconnect=on_disconnect,
            conf_threshold=args.conf,
            show_video=not args.no_gui,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        listener.close()


if __name__ == "__main__":
    # Ensure we can import the client module when running from different working dirs
    sys.exit(main())
