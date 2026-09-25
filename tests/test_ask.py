from ask import evidence_supported, tokenize

SOURCE = (
    "For the PvP ruleset, faction balance remains an important consideration. "
    "On the PvP ruleset, players can only create characters for one faction. "
    "Once you create an Alliance character on the PvP ruleset, you won’t be able "
    "to create a Horde character there, and vice versa."
)
HITS = [(SOURCE, {"title": "Choosing Your Ruleset"})]


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What is the Level Cap?") == ["level", "cap"]


def test_tokenize_keeps_apostrophes_inside_names():
    assert "ula'tek" in tokenize("Nerfs to Ula'tek on Mythic")


def test_tokenize_treats_curly_and_straight_apostrophes_alike():
    # Wowhead writes Ula’tek, users type Ula'tek; BM25 has to see the same token
    assert tokenize("Nerfs to Ula’tek") == tokenize("Nerfs to Ula'tek")


def test_exact_quote_is_supported():
    assert evidence_supported(["On the PvP ruleset, players can only create characters for one faction."], HITS)


def test_case_and_apostrophe_style_are_ignored():
    quote = (
        "ONCE YOU CREATE an Alliance character on the PvP ruleset, "
        "you won't be able to create a Horde character there, and vice versa."
    )
    assert evidence_supported([quote], HITS)


def test_fabricated_sentence_is_rejected():
    assert not evidence_supported(["Players can freely create characters of both factions on every ruleset."], HITS)


def test_paraphrase_is_rejected():
    assert not evidence_supported(["On the PvP ruleset, people may only make characters for a single faction."], HITS)


def test_short_fragment_does_not_count_even_if_present():
    assert not evidence_supported(["one faction."], HITS)


def test_merged_sentences_pass_if_one_is_verbatim():
    # real case: the model glued two sentences together and swapped "players" for "you" in the first
    reworded = "On the PvP ruleset, you can only create characters for one faction."
    verbatim = (
        "Once you create an Alliance character on the PvP ruleset, you won’t be able "
        "to create a Horde character there, and vice versa."
    )
    assert not evidence_supported([reworded], HITS)
    assert evidence_supported([f"{reworded} {verbatim}"], HITS)


def test_empty_evidence_is_rejected():
    assert not evidence_supported([], HITS)


def test_quote_must_come_from_the_retrieved_hits():
    other_hits = [("Seal of Fury grants a small absorb shield and lets Judgment taunt.", {})]
    assert not evidence_supported(
        ["On the PvP ruleset, players can only create characters for one faction."], other_hits
    )