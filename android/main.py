from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.webview import WebView
from kivy.clock import Clock
import threading


class AWEAIApp(App):
    """Android launcher: starts the local AWEAI UI and shows it full-screen."""

    def build(self):
        layout = BoxLayout()
        self.webview = WebView(url="http://127.0.0.1:8888")
        layout.add_widget(self.webview)
        threading.Thread(target=self._start_server, daemon=True).start()
        return layout

    def _start_server(self):
        # Local UI server: port 8888 (auto-increments if busy)
        from aweai.ui.api import serve

        serve(host="127.0.0.1", open_browser=False)


if __name__ == "__main__":
    AWEAIApp().run()
