"""AWEAI Model Factory — Android entrypoint.

Starts the local factory UI server and opens it in a WebView-style browser.
Kept intentionally tiny; full functionality comes from the server.
"""
import threading
import webbrowser


def main() -> None:
    from aweai.ports import resolve_port
    from aweai.ui import serve

    port = resolve_port(8888)
    print(f"AWEAI Android — starting factory UI on port {port}")
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    serve(port=port, host="0.0.0.0", open_browser=False)


if __name__ == "__main__":
    main()
