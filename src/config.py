import os
from pathlib import Path

NEWS = "forever-news"
CLASSIC = "classic-reference"

FEEDS = [
    "https://www.wowhead.com/news/rss/all",
]

SEED_URLS = [
    "https://news.blizzard.com/en-us/article/24303313/world-of-warcraft-forever-deep-dive-panel-recap",
    "https://news.blizzard.com/en-us/article/24302093/carve-a-new-path-with-world-of-warcraft-forever",
    "https://news.blizzard.com/en-us/article/24302498/pre-purchase-the-world-of-warcraft-forever-collectors-edition",
    "https://news.blizzard.com/en-us/article/24301508/pre-purchase-world-of-warcraft-forever-upgrades-and-begin-your-next-journey-in-azeroth",
    "https://news.blizzard.com/en-us/article/24303862/world-of-warcraft-forever-whats-next-panel-recap",
    "https://news.blizzard.com/en-us/article/24303312/submit-your-questions-for-the-world-of-warcraft-live-q-a-september-17",
    "https://news.blizzard.com/en-us/article/24304071/world-of-warcraft-forever-found-photos-panel-recap",
    "https://news.blizzard.com/en-us/article/24302071/wow-forever-meet-the-new-skyborne-guardians",
    "https://news.blizzard.com/en-us/article/24304075/create-the-hero-you-want-to-be-in-world-of-warcraft-forever",
    "https://news.blizzard.com/en-us/article/24304161/create-a-name-of-your-own-in-wow-forever",
    "https://news.blizzard.com/en-us/article/24302070/choose-your-ruleset-in-world-of-warcraft-forever",
    "https://news.blizzard.com/en-us/article/24307383/get-to-know-the-world-of-warcraft-forever-legacy-system",
    "https://news.blizzard.com/en-us/article/24304071/world-of-warcraft-forever-found-photos-panel-recap",
]

CLASSIC_SEED_URLS = [
    "https://www.warcrafttavern.com/wow-classic/guides/compendium/",
    "https://news.blizzard.com/en-us/article/23090134/wow-classic-primer-for-new-players",
    "https://warcraft.wiki.gg/wiki/World_of_Warcraft:_Classic",
]

KEYWORDS = ["warcraft forever", "wow forever", "classic+", "classic plus"]

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw"
USER_AGENT = "wow-forever-rag/0.1 (personal research project)"
REQUEST_DELAY_SECONDS = 1.0
