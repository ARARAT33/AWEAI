"""Example: memory store persistence demo.

Run:
    python examples/memory_demo.py
"""

import os
import tempfile

from aweai.memory import MemoryStore


def main() -> None:
    path = os.path.join(tempfile.gettempdir(), "aweai_memory_demo.db")
    store = MemoryStore(path)
    try:
        store.add_message("user", "Remember that I like Armenian coffee.")
        store.add_message("assistant", "Noted!")
        store.set("favorite_drink", "Armenian coffee")

        print("Messages:")
        for row in store.get_messages():
            print(f"  [{row['role']}] {row['content']}")
        print("favorite_drink =", store.get("favorite_drink"))
        print("Search 'coffee':")
        for row in store.search("coffee"):
            print(f"  [{row['role']}] {row['content']}")
    finally:
        store.close()
        os.remove(path)


if __name__ == "__main__":
    main()
