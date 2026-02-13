from streaming.stream_server import StreamServer

if __name__ == "__main__":
    srv = StreamServer()
    try:
        srv.connect()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        srv.close()