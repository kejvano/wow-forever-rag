import fetch
import index

from datetime import datetime, timezone

if __name__ == "__main__":
    print(f"\n=== update {datetime.now(timezone.utc).isoformat()} ===")

    fetch.run()
    index.run()