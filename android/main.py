"""AWEAI Android launcher.

Starts the AWEAI UI server locally (on a free port starting at 8888) and
displays it in a full-screen Kivy WebView. This is the entry point used by
buildozer when building the APK.
"""

import os
import sys
import threading

# Make sure the package is importable from the bundled path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.webview import WebView  # WebView is available via pyjnius/android


class AWEAIApp(App):
    def build(self):
        layout = BoxLayout(orientation="vertical")
        self.status = Label(
            text="Starting AWEAI…", size_hint_y=0.08, font_size="16sp"
        )
        layout.add_widget(self.status)
        self.webview = WebView(url="about:blank")
        layout.add_widget(self.webview)
        threading.Thread(target=self._start_server, daemon=True).start()
        return layout

    def _start_server(self):
        from aweai.ports import resolve_port
        from aweai.ui.api import create_app
        import uvicorn

        try:
            port = resolve_port(8888)
            app = create_app()
            # uvicorn in a thread on Android; the WebView loads the local URL
            self.webview.url = f"http://127.0.0.1:{port}/"
            self.status.text = f"AWEAI — http://127.0.0.1:{port}"
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
        except Exception as e:
            self.status.text = f"Error: {e}"


if __name__ == "__main__":
    AWEAIApp().run()
