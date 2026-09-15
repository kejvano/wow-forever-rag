import fetch
import index
import requests

from datetime import datetime, timezone

if __name__ == "__main__":
    print(f"\n=== update {datetime.now(timezone.utc).isoformat()} ===")

    fetch.run()
    index.run()

    try:
        requests.post("http://127.0.0.1:8000/reload", timeout=10)
    except requests.RequestException:
        pass  # server not running, index will load on next start