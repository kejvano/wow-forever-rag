from ask import REFUSAL
from evaluate import all_found, check


def test_plain_string_must_appear():
    assert all_found("50 gold", "costs up to 50 gold")
    assert not all_found("50 gold", "costs up to 10 gold")


def test_list_means_every_fragment_is_required():
    assert all_found(["20", "30"], "starts at 20, rises to 30")
    assert not all_found(["20", "30"], "the cap is 30")


def test_nested_list_means_any_of():
    assert all_found([["one faction", "Horde"]], "you can't create a Horde character there")
    assert not all_found([["one faction", "Horde"]], "no restrictions at all")


def test_matching_ignores_case():
    assert all_found("undead paladin", "Undead Paladin is a new combination")


def test_empty_expectation_always_passes():
    assert all_found("", "anything")
    assert all_found([], "anything")


def test_expect_no_background_fails_when_background_present():
    got = {"answer": REFUSAL, "background": "In Classic, Cooking was a secondary skill."}
    assert not check({"expect_refusal": True, "expect_no_background": True}, got)


def test_refusal_case_fails_on_a_real_answer():
    got = {"answer": "The cap is 60.", "background": ""}
    assert not check({"expect_refusal": True}, got)