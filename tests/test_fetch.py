from fetch import fallback_metadata, strip_trailing_nav


def test_strip_trailing_nav_cuts_related_links_block():
    text = "Real article text.\nMore text.\nBlizzCon 2026 Day 2\nWarcraft: Forever\nOver 600 New Recipes"
    assert strip_trailing_nav(text) == "Real article text.\nMore text."


def test_strip_trailing_nav_handles_heading_without_day():
    assert strip_trailing_nav("Article.\nBlizzCon 2026\nNew Zones") == "Article."


def test_strip_trailing_nav_ignores_mentions_inside_sentences():
    text = "Forever was announced at BlizzCon 2026 in September."
    assert strip_trailing_nav(text) == text


def test_fallback_metadata_survives_broken_json_ld():
    # Blizzard's structured data is missing a comma, so it can't be parsed as JSON
    html = """<html><head>
    <meta property="og:title" content="World of Warcraft: Forever Deep Dive Panel Recap"/>
    <script type="application/ld+json">{
      "datePublished": "2026-09-13T17:52:40Z",
      "author": [{"name": "Blizzard Entertainment"}]
      "publisher": [{"name": "Blizzard Entertainment"}]
    }</script>
    </head><body></body></html>"""
    title, published = fallback_metadata(html)
    assert title == "World of Warcraft: Forever Deep Dive Panel Recap"
    assert published == "2026-09-13T17:52:40Z"


def test_fallback_metadata_without_any_metadata():
    assert fallback_metadata("<html><body>nothing here</body></html>") == ("", None)