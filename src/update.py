from datetime import datetime, timezone

import fetch
import index


def run() -> None:
    print(f"\n=== update {datetime.now(timezone.utc).isoformat()} ===", flush=True)
    fetch.run()
    index.run()


if __name__ == "__main__":
    run()
