"""Offline regressions for the gated extractor's evidence checks and local rules. No provider access."""

import json

import pytest

from live_form_eval import Partial, load_config
from live_form_eval.data.common.store import load_dataset
from strategies.delta import parse_delta
from strategies.services import requested_services
from strategies.spelling import SpelledName


@pytest.fixture(scope="module")
def form():
    data = load_dataset(load_config(), "spokenwoz", official_sample=True)
    return next(iter(data.forms.values()))


def op(field_id, value, quote):
    return json.dumps({"set": [{"field": field_id, "value": value, "quote": quote}], "facts": []})


def delta(form, field_id, value, quote, *sources, current=None):
    return parse_delta(form, current or {}, op(field_id, value, quote), evidence_sources=sources)


# British and American spellings -------------------------------------------------------------------------------

@pytest.mark.parametrize("field_id,value,text", [
    ("restaurant_area", "centre", "A: somewhere in the center of the city"),
    ("hotel_area", "centre", "A: a hotel near the city center please"),
    ("attraction_type", "theatre", "A: is there a theater I could visit"),
    ("attraction_area", "centre", "A: something in the centre"),
])
def test_spelling_variants_ground(form, field_id, value, text):
    quote = text.removeprefix("A: ")
    assert delta(form, field_id, value, quote, text) == {field_id: value}


def test_spelling_variant_needs_the_word(form):
    assert delta(form, "restaurant_area", "centre", "in the city", "A: somewhere in the city") == {}


@pytest.mark.parametrize("value,text", [
    ("south", "A: it should be on the opposite side of of the north"),
    ("east", "A: the opposite end of town from the west"),
])
def test_opposite_direction_grounds(form, value, text):
    quote = text.removeprefix("A: ")
    assert delta(form, "restaurant_area", value, quote, text) == {"restaurant_area": value}


def test_opposite_needs_the_other_direction(form):
    text = "A: the opposite side of the centre"
    assert delta(form, "restaurant_area", "south", text[3:], text) == {}


# Group sizes --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("field_id,quote,value", [
    ("restaurant_people", "myself and five other people", 6),
    ("hotel_people", "me and two others", 3),
    ("train_people", "myself plus 3 more", 4),
    ("restaurant_people", "for me and one other person", 2),
    ("restaurant_people", "myself and five", 6),   # a quote cut before "others"
])
def test_caller_counts_in_group(form, field_id, quote, value):
    assert delta(form, field_id, value, quote, f"A: {quote}") == {field_id: value}


def test_companion_count_is_not_group_size(form):
    quote = "myself and five other people"
    assert delta(form, "restaurant_people", 5, quote, f"A: {quote}") == {}


def test_plain_group_numbers_still_ground(form):
    assert delta(form, "restaurant_people", 5, "five of us", "A: there are five of us") == {"restaurant_people": 5}
    assert delta(form, "restaurant_people", 6, "a table for six", "A: a table for six") == {"restaurant_people": 6}


# Relative days ------------------------------------------------------------------------------------------------

SNG1193_DAY = "A: at quarter past 11, today is a thursday or two days from now."


def test_reference_day_is_not_booking_day(form):
    assert delta(form, "restaurant_day", "thursday", "today is a thursday", SNG1193_DAY) == {}


def test_offset_from_reference_day(form):
    quote = "today is a thursday or two days from now"
    assert delta(form, "restaurant_day", "saturday", quote, SNG1193_DAY) == {"restaurant_day": "saturday"}
    assert delta(form, "restaurant_day", "friday", quote, SNG1193_DAY) == {}


def test_offset_quote_uses_reference_in_window(form):
    assert delta(form, "restaurant_day", "saturday", "two days from now", SNG1193_DAY) == {
        "restaurant_day": "saturday"}


@pytest.mark.parametrize("text,value", [
    ("A: it's monday today, I need a room from tomorrow", "tuesday"),
    ("A: today is friday. we arrive the day after tomorrow", "sunday"),
    ("A: today's saturday and I want to travel in three days", "tuesday"),
    ("A: today is sunday, book it for 1 day from now", "monday"),
])
def test_synthetic_relative_days(form, text, value):
    quote = text.removeprefix("A: ")
    field_id = "hotel_day" if "room" in text or "arrive" in text else "train_day"
    assert delta(form, field_id, value, quote, text) == {field_id: value}


def test_reference_day_without_offset_can_be_the_day(form):
    text = "A: today is thursday and I'd like a table for tonight"
    assert delta(form, "restaurant_day", "thursday", "today is thursday", text) == {"restaurant_day": "thursday"}


def test_restated_day_still_grounds(form):
    text = "A: today is thursday, two days from now. B: so on saturday"
    assert delta(form, "restaurant_day", "saturday", "on saturday", text) == {"restaurant_day": "saturday"}
    text = "A: today is monday, tomorrow is fine but actually make it monday next week"
    assert delta(form, "hotel_day", "monday", "make it monday next week", text) == {"hotel_day": "monday"}


# Service requests ---------------------------------------------------------------------------------------------

def turn(text, speaker="A"):
    return Partial(t=1, kind="turn", turn_id="1", text=text, speaker=speaker)


@pytest.mark.parametrize("text", [
    "could you please book a table for myself and five other people.",
    "I'd like to reserve a table at eight.",
    "can you get us a table near the river?",
    "i'm looking for a place that we can eat indian food in the city.",
    "we need somewhere to have dinner tonight.",
])
def test_book_a_table_requests_restaurant(form, text):
    assert requested_services(form, turn(text), caller="A") == {"restaurant"}


@pytest.mark.parametrize("text,speaker", [
    ("i can book a table for you.", "B"),        # the agent offering
    ("the room has a table and a desk.", "A"),   # furniture
    ("i want a table lamp.", "A"),
    ("is there a place to eat near the hotel?", "A"),   # a question, not a request
])
def test_table_without_request(form, text, speaker):
    assert requested_services(form, turn(text, speaker), caller="A") == set()


@pytest.mark.parametrize("text,expected", [
    ("uh, looking for train to london, liverpool.", {"train"}),
    ("perfect. i'm looking for a place to stay in the center of cambridge.", {"hotel"}),
    ("we need somewhere to stay for two nights.", {"hotel"}),
    ("is there a place to stay near the station?", set()),   # a question
    ("i'm looking forward to it.", set()),
])
def test_service_requests_without_a_subject_or_the_word(form, text, expected):
    assert requested_services(form, turn(text), caller="A") == expected


# Recognizer misspellings of an option -------------------------------------------------------------------------

def test_misheard_option_grounds_when_one_option_is_close(form):
    text = "A: uh, it should depart from kingsland and it should, um. head to cambridge."
    assert delta(form, "train_departure", "kings lynn", "depart from kingsland", text) == {
        "train_departure": "kings lynn"}
    assert delta(form, "train_departure", "ely", "depart from kingsland", text) == {}


def test_short_or_distant_options_are_not_guessed(form):
    text = "A: something in the yeast please"
    assert delta(form, "restaurant_area", "east", "something in the yeast", text) == {}
    text = "A: in the exclusive price range"
    assert delta(form, "restaurant_price_range", "expensive", "exclusive price range", text) == {}


def test_restaurant_service_grounds_on_eating_request(form):
    text = "A: i'm looking for a place that we can eat indian food"
    assert delta(form, "services", ["restaurant"], "we can eat indian food", text) == {"services": ["restaurant"]}
    assert delta(form, "services", ["hotel"], "we can eat indian food", text) == {}


def test_name_parts_said_apart_ground(form):
    text = "A: certainly. so first name is rebecca. my last name is louie."
    quote = "first name is rebecca. my last name is louie"
    assert delta(form, "profile_name", "rebecca louie", quote, text) == {"profile_name": "rebecca louie"}
    assert delta(form, "profile_name", "louie rebecca", quote, text) == {}
    assert delta(form, "profile_name", "rebecca lothridge", quote, text) == {}


# Spelled names ------------------------------------------------------------------------------------------------

def spell(turns, current):
    tracker, result = SpelledName(), None
    for speaker, text in turns:
        result = tracker.observe(speaker, text, current) or result
    return result or tracker.finish(current)


def test_sng1193_spelled_surname():
    turns = [
        ("A", "certainly. so first name is rebecca. my last name is louie."),
        ("B", "okay, alright, thank you may please spell your last name for me."),
        ("A", "mhm. that's both as l like lima or orange."), ("B", "mhm. yes."), ("A", "she likes to."),
        ("B", "uh huh."), ("A", "age like hotel. are like rojak."), ("B", "yes."),
        ("A", "i like india. do you like delta."), ("B", "uh huh."), ("A", "do you like golf."),
        ("B", "what."), ("A", "he like echo."),
        ("B", "okay, all right. thank you very much. and so it has been booked for you."),
    ]
    assert spell(turns, "rebecca louie") == "rebecca lothridge"


@pytest.mark.parametrize("turns,current,expected", [
    # NATO words with "as in" and "for"
    ([("B", "could you spell your surname?"), ("A", "k as in kilo, a as in alpha, h as in hotel, n as in november"),
      ("B", "thank you, and your phone number?")], "james con", "james kahn"),
    # Plain letters
    ([("B", "can you spell the last name"), ("A", "sure, p e r e z"), ("B", "great, that is booked")],
     "maria peres", "maria perez"),
    # A first name being spelled
    ([("B", "how do you spell your first name"), ("A", "s for sierra e for echo a for alpha n for november"),
      ("B", "thanks, booking that now")], "shawn lee", "sean lee"),
])
def test_synthetic_spellings(turns, current, expected):
    assert spell(turns, current) == expected


def test_spelling_that_confirms_name_changes_nothing():
    # MUL2873: a garbled spelling of "edward smith" must not replace the correct name.
    turns = [
        ("B", "edward smith. may you please spell it for me."),
        ("A", ("okay, my first name is paul is e dw ard. iceland name is ff for sierra m for mike l. for india, "
               "t for tango h for.")),
        ("B", "okay, thank you. would you like an email order."),
    ]
    assert spell(turns, "edward smith") is None


def test_spelling_checked_against_heard_name_when_form_is_empty():
    # Same call with the model silent: the caller's own "my full name is edward smith" is the reference.
    turns = [
        ("A", "hi, this is edward speaking."), ("A", "certainly my full name is edward smith."),
        ("B", "edward smith. may you please spell it for me."),
        ("A", ("my first name is paul is e dw ard. iceland name is ff for sierra m for mike l. for india, "
               "t for tango h for.")),
        ("B", "okay, thank you. would you like an email order."),
    ]
    assert spell(turns, None) is None


def test_spelled_surname_with_model_silent():
    turns = [
        ("A", "first name is rebecca. my last name is louie."), ("B", "please spell your last name"),
        ("A", "l like lima, o like oscar, t like tango, h like hotel"), ("B", "thank you, that is booked"),
    ]
    assert spell(turns, None) == "rebecca loth"


def test_spelling_that_drops_letters_keeps_name():
    # Known limit: "jon" fits inside "john", so it reads as a partial confirmation.
    turns = [("B", "how do you spell your first name"), ("A", "j for juliet o for oscar n for november"),
             ("B", "thanks")]
    assert spell(turns, "john lee") is None


def test_no_spelling_without_request():
    assert spell([("A", "i like india. do you like delta."), ("B", "sure, booked.")], "ann lee") is None


def test_too_few_letters_publish_nothing():
    turns = [("B", "please spell your last name"), ("A", "l like lima"), ("B", "sorry, the line is bad")]
    assert spell(turns, "rebecca louie") is None


def test_spelling_variant_value_normalizes_to_option(form):
    text = "A: in the center of the city"
    assert delta(form, "restaurant_area", "center", "center of the city", text) == {"restaurant_area": "centre"}


def test_dropped_proposals_are_logged(form):
    diagnostics = {}
    parse_delta(form, {}, op("restaurant_time", "11:15", "at noon"), evidence_sources=("A: at noon",),
                diagnostics=diagnostics)
    assert diagnostics["dropped"][0]["reason"] == "value_not_in_quote"


def test_unmarked_hour_is_shown_as_said(form):
    text = "A: at quarter past 11, today is a thursday"
    assert delta(form, "restaurant_time", "23:15", "quarter past 11", text) == {"restaurant_time": "11:15"}
    text = "A: at quarter past 11 at night"
    assert delta(form, "restaurant_time", "23:15", "quarter past 11 at night", text) == {"restaurant_time": "23:15"}
    assert delta(form, "restaurant_time", "11:15", "quarter past 11 at night", text) == {"restaurant_time": "23:15"}
    text = "A: at 7 pm please"
    assert delta(form, "restaurant_time", "19:00", "at 7 pm", text) == {"restaurant_time": "19:00"}
    assert delta(form, "restaurant_time", "07:00", "at 7 pm", text) == {"restaurant_time": "19:00"}


def test_gate_sees_spelling_variant_as_option(form):
    from strategies.gate import Gate
    schedule = Gate(form).schedule("in the center of the city", recent="", first_at=10, last_call=0,
                                   finished=False, final=False)
    assert schedule is not None and schedule.reason == "slot_signal"


# Found in the full Gemini 2.5 Flash run ----------------------------------------------------------------------

def test_dictated_id_across_backchannels(form):
    text = "A: yeah, it's 0, 3, 0, 3. B: okay. A: 7 8 2 3. B: okay. A: 1 7 7 9."
    quote = "0, 3, 0, 3. 7 8 2 3. 1 7 7 9."   # the model left out the agent's "okay"
    assert delta(form, "profile_id_number", "030378231779", quote, text) == {"profile_id_number": "030378231779"}
    assert delta(form, "profile_id_number", "0303", "my id is 0, 3, 0, 3.", "A: my id is 0, 3, 0, 3.") == {
        "profile_id_number": "0303"}
    assert delta(form, "profile_id_number", "030378231780", quote, text) == {}


def test_two_word_prefix_names_long_station(form):
    text = "A: i need a train to london, liverpool"
    assert delta(form, "train_destination", "london liverpool street", "train to london, liverpool", text) == {
        "train_destination": "london liverpool street"}
    # "london kings" is shared by no other option, but "london" alone is ambiguous.
    assert delta(form, "train_destination", "london kings cross", "train to london", "A: train to london") == {}


def test_partner_makes_two(form):
    text = "A: it will be myself and my partner"
    assert delta(form, "hotel_people", 2, "myself and my partner", text) == {"hotel_people": 2}


def test_doesnt_need_parking_is_false(form):
    text = "A: it doesn't need to have free parking"
    assert delta(form, "hotel_parking", False, "it doesn't need to have free parking", text) == {"hotel_parking": False}


def test_attraction_service_cues(form):
    text = "A: i also looking for places to go in town"
    assert delta(form, "services", ["attraction"], "looking for places to go in town", text) == {
        "services": ["attraction"]}


def test_http_errors_and_single_timeouts_do_not_stop_the_run(monkeypatch):
    import asyncio

    from strategies import openrouter

    client = openrouter.OpenRouterClient(api_key="test", model="google/gemini-2.5-flash",
                                         usd_per_million={"input_tokens": 0.3, "output_tokens": 2.5},
                                         eur_per_usd=0.86, budget_usd=1.0)
    calls = []

    def fail(messages, max_tokens):
        calls.append(1)
        error = openrouter._RequestFailed("OpenRouter returned HTTP 503.")
        error.http_status = 503
        raise error

    monkeypatch.setattr(client, "_request", fail)
    messages = [{"role": "user", "content": "x"}]
    first = asyncio.run(client.complete(messages, max_tokens=50))
    second = asyncio.run(client.complete(messages, max_tokens=50))
    assert first.error and second is not None and len(calls) == 2

    def hang(messages, max_tokens):
        raise TimeoutError

    monkeypatch.setattr(client, "_request", hang)
    spent = client._spent_usd
    timed_out = asyncio.run(client.complete(messages, max_tokens=50))
    assert timed_out.error and client._spent_usd > spent  # a timeout may bill: its reserve counts as spent

    monkeypatch.setattr(client, "_request", lambda messages, max_tokens: {
        "model": "google/gemini-2.5-flash", "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.00001},
        "choices": [{"message": {"content": '{"set":[],"facts":[]}'}}]})
    recovered = asyncio.run(client.complete(messages, max_tokens=50))
    assert recovered is not None and recovered.error is None  # one timeout does not stop the run

    monkeypatch.setattr(client, "_request", hang)
    for _ in range(openrouter.MAX_UNRESOLVED):
        asyncio.run(client.complete(messages, max_tokens=50))
    assert asyncio.run(client.complete(messages, max_tokens=50)) is None  # repeated timeouts stop it


def test_dictated_number_extends_digits_already_shown(form):
    # The first groups scrolled out of the window; the model quotes the new group only.
    text = "A: 1, 4, 3, 2."
    current = {"profile_id_number": "030378231779"}
    assert delta(form, "profile_id_number", "0303782317791432", "1, 4, 3, 2.", text, current=current) == {
        "profile_id_number": "0303782317791432"}
    assert delta(form, "profile_id_number", "0303782317791433", "1, 4, 3, 2.", text, current=current) == {}


def test_dictation_spans_recent_and_new_turns(form):
    recent, new = "A: 0, 3, 0, 3. B: okay.", "A: 7 8 2 3."
    quote = "0, 3, 0, 3. 7 8 2 3."
    assert parse_delta(form, {}, op("profile_id_number", "03037823", quote),
                       evidence_sources=(new, "", recent)) == {"profile_id_number": "03037823"}


def test_stated_full_name_is_taken_once():
    tracker = SpelledName()
    tracker.observe("A", "hi, my name is mary edwards.", None)
    assert tracker.take_stated() == "mary edwards" and tracker.take_stated() is None
    for text in ("my name is not important", "first name is rebecca. my last name is louie.",
                 "you are speaking to rebecca lottery"):
        other = SpelledName()
        other.observe("A", text, None)
        assert other.take_stated() is None, text
    agent = SpelledName()
    agent.observe("B", "my name is john smith, how can i help", None)
    assert agent.take_stated() is None


@pytest.fixture(scope="module")
def meeting():
    data = load_dataset(load_config(), "ami", official_sample=True)
    return next(iter(data.forms.values()))


def test_meeting_topics_need_a_quote_not_the_topic_name(meeting):
    text = "B: I'm first going to do an opening then we get used to one another"
    reply = json.dumps({"set": [{"field": "topics", "value": ["opening", "chitchat"],
                                 "quote": "I'm first going to do an opening"}], "facts": []})
    assert parse_delta(meeting, {}, reply, evidence_sources=(text,)) == {"topics": ["opening", "chitchat"]}
    reply = json.dumps({"set": [{"field": "topics", "value": ["opening"], "quote": "we discussed openings"}],
                        "facts": []})
    assert parse_delta(meeting, {}, reply, evidence_sources=(text,)) == {}


def test_meeting_topics_sent_as_facts_are_selected(meeting):
    text = "C: So she likes uh buttons which make click"
    reply = json.dumps({"set": [], "facts": [{"field": "topics", "add": ["user requirements", "not a topic"],
                                              "remove": [], "quote": "she likes uh buttons which make click"}]})
    assert parse_delta(meeting, {"topics": ["opening"]}, reply, evidence_sources=(text,)) == {
        "topics": ["opening", "user requirements"]}


def test_whole_note_as_value_becomes_fact_lines(meeting):
    field = next(spec.field_id for spec in meeting.fields if spec.type.value == "textarea")
    text = "C: So they have to be waterproof maybe"
    reply = json.dumps({"set": [{"field": field, "value": "The Marketing Expert said it should be waterproof.",
                                 "quote": "they have to be waterproof"}], "facts": []})
    assert parse_delta(meeting, {}, reply, evidence_sources=(text,)) == {
        field: "The Marketing Expert said it should be waterproof."}


def test_meeting_quote_across_speakers(meeting):
    field = next(spec.field_id for spec in meeting.fields if spec.type.value == "textarea")
    sources = ("C: Uh yes, we have plenty at home. In fact, my daughter likes l remote controls.",
               "B: So that could be a great um application.")
    quote = "C: Uh yes, we have plenty at home. In fact, my daughter likes l remote controls. B: So that could be"
    reply = json.dumps({"set": [], "facts": [{"field": field, "add": ["Her daughter likes remote controls."],
                                              "remove": [], "quote": quote}]})
    assert parse_delta(meeting, {}, reply, evidence_sources=sources) == {field: "Her daughter likes remote controls."}
    made_up = json.dumps({"set": [], "facts": [{"field": field, "add": ["x"], "remove": [],
                                                "quote": "we agreed the remote should be yellow and round"}]})
    assert parse_delta(meeting, {}, made_up, evidence_sources=sources) == {}


def test_meeting_quote_running_past_a_revised_word(meeting):
    field = next(spec.field_id for spec in meeting.fields if spec.type.value == "textarea")
    text = "B: we're going to meet again in thirty minutes and uh I want you to mm work on these ideas"
    reply = json.dumps({"set": [], "facts": [{"field": field, "add": ["Meet again in thirty minutes."], "remove": [],
                                              "quote": "we're going to meet again in thirty minutes and uh I want "
                                                       "you to mm work on these ideas and try to make a uh the"}]})
    assert parse_delta(meeting, {}, reply, evidence_sources=(text,)) == {field: "Meet again in thirty minutes."}


def test_several_meeting_notes_for_one_field_in_one_reply(meeting):
    field = next(spec.field_id for spec in meeting.fields if spec.type.value == "textarea")
    text = "D: No screens, it's too complex. A: Too expensive, yeah."
    reply = json.dumps({"set": [], "facts": [
        {"field": field, "add": ["No screens, too complex."], "remove": [], "quote": "No screens, it's too complex."},
        {"field": field, "add": ["A screen is too expensive."], "remove": [], "quote": "Too expensive, yeah."}]})
    assert parse_delta(meeting, {}, reply, evidence_sources=(text,)) == {
        field: "No screens, too complex.\nA screen is too expensive."}


def test_quote_with_one_corrected_word(form):
    text = "A: uh, i'd like to live the place of attraction by 11 a. m.."
    assert delta(form, "taxi_leave_at", "11:00", "leave the place of attraction by 11 a. m.", text) == {
        "taxi_leave_at": "11:00"}
    # A changed number is never a correction of the recognizer.
    assert delta(form, "taxi_leave_at", "12:00", "live the place of attraction by 12 a. m.", text) == {}
    # Short quotes must match exactly.
    assert delta(form, "taxi_leave_at", "11:00", "leave at 11", "A: live at 11") == {}


def test_agent_information_is_not_a_caller_value(form):
    text = "A: they address the phone number and the full type. B: okay, the restaurant serves italian food."
    assert delta(form, "restaurant_food", "italian", "the restaurant serves italian food", text) == {}
    text = ("A: we can eat indian food in the center. B: that would be the golden curry. and they definitely serve "
            "indian. is that okay with you.")
    assert delta(form, "restaurant_name", "golden curry", "golden curry", text) == {"restaurant_name": "golden curry"}
    text = "A: it should depart from kingsland. B: from kings lynn to cambridge, right."
    assert delta(form, "train_departure", "kings lynn", "from kings lynn", text) == {"train_departure": "kings lynn"}
    text = "A: a restaurant serving italian food please"
    assert delta(form, "restaurant_food", "italian", "italian food", text) == {"restaurant_food": "italian"}


def test_agent_answer_across_sources(form):
    recent, new = "A: the entry fee open hours in the area.", "B: and the cambridge corn exchange is located in the centre."
    reply = op("attraction_area", "centre", "located in the centre")
    assert parse_delta(form, {}, reply, evidence_sources=(new, "", recent)) == {}


def test_agent_proposal_after_caller_constraints_stays(form):
    text = ("A: we can eat indian food in the exclusive price range in the center of the city. "
            "B: let me see the sense of the city that would be the golden curry.")
    assert delta(form, "restaurant_name", "golden curry", "the golden curry", text) == {
        "restaurant_name": "golden curry"}


# Dictated numbers ---------------------------------------------------------------------------------------------

from strategies.dictation import DigitDictation  # noqa: E402

_NUMBERS = {"profile_phone", "profile_id_number"}


def _dictate(turns):
    dictation = DigitDictation(_NUMBERS)
    shown = {}
    for speaker, text in turns:
        if found := dictation.observe(speaker, text):
            shown[found[0]] = found[1]
    return dictation, shown


def test_id_dictated_in_groups_with_agent_echo():
    _, shown = _dictate([
        ("B", "okay, i'll be happy to do that. may i please have your id number."),
        ("A", "sure, my id number is 0, 2, 3, 8."), ("B", "yeah."), ("A", "9 2 2 1. 0, 3, 2, 8."),
        ("B", "92210328."), ("A", "2, 4, 8, 8."),
        ("B", "okay, thank you very much. train booking for two has been made and your train id is tr8104."),
    ])
    assert shown == {"profile_id_number": "0238922103282488"}


def test_ask_in_a_turn_that_also_gives_prices():
    _, shown = _dictate([
        ("B", "so the train id for that will be tr4890. and there'll be 13, £28 per person. could i get your id "
              "number to make the booking, please."),
        ("A", "yeah, it's 0, 3, 0, 3."), ("B", "okay."), ("A", "7 8 2 3."),
    ])
    assert shown == {"profile_id_number": "03037823"}


def test_caller_announces_phone_with_double():
    dictation, shown = _dictate([
        ("B", "all right, ma'am, she says it's telephone number."),
        ("A", "my telephone number is 2 4 3 7 4 7. uh, double 1 7 8."),
        ("B", "okay, just to confirm, they said the cellphone number is 2437471178. is that correct."),
        ("A", "yes, that is correct."), ("B", "so is there anything else i should like me to assist you."),
    ])
    assert shown == {"profile_phone": "2437471178"}
    assert dictation.claims("profile_id_number", "1178") and not dictation.claims("profile_id_number", "5550")


def test_agent_giving_a_number_is_not_a_dictation():
    _, shown = _dictate([
        ("A", "can i have the postal code and phone number."),
        ("B", "sure. so the postal code is cb23ht. and the phone number is 1223324600."),
        ("A", "okay, thank you."),
    ])
    assert shown == {}


def test_partial_digits_show_early():
    dictation = DigitDictation(_NUMBERS)
    dictation.observe("B", "could i have your phone number please")
    assert dictation.observe("A", "it's 0 7 7", partial=True) == ("profile_phone", "077")
    assert dictation.observe("A", "it's 0 7 7 1") == ("profile_phone", "0771")
    assert dictation.claims("profile_phone", "0771")  # still dictating: the model waits


# Agent speech does not replace a caller's value -----------------------------------------------------------------

def test_agent_train_time_does_not_replace_caller_constraint(form):
    sources = ("B: 9 17 a. m.", "", "A: anything that would arrive in london by 9 45 a m would be suitable.")
    current = {"train_arrive_by": "09:45"}
    assert delta(form, "train_arrive_by", "09:17", "9 17 a. m.", *sources, current=current) == {}
    assert delta(form, "train_arrive_by", "09:17", "9 17 a. m.", *sources) == {"train_arrive_by": "09:17"}


def test_agent_readback_the_caller_confirms_can_replace(form):
    sources = ("B: sorry, my bad, cambridge to london liverpool street. is that correct. A: yes, that's right.", "",
               "A: from london liverpool street to cambridge.")
    current = {"train_departure": "london liverpool street"}
    assert delta(form, "train_departure", "cambridge", "cambridge to london liverpool street", *sources,
                 current=current) == {"train_departure": "cambridge"}


def test_caller_correction_still_replaces(form):
    sources = ("A: actually make it the expensive range.", "", "A: moderate price range")
    current = {"restaurant_price_range": "moderate"}
    assert delta(form, "restaurant_price_range", "expensive", "make it the expensive range", *sources,
                 current=current) == {"restaurant_price_range": "expensive"}


def test_system_prompt_is_sent_with_a_cache_breakpoint(monkeypatch):
    from strategies import openrouter

    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "{}"}]
    cached = openrouter._cacheable(messages)
    assert cached[0]["content"] == [{"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}]
    assert cached[1] == messages[1] and messages[0]["content"] == "rules"  # the caller's list is unchanged
    monkeypatch.setenv("LFE_OPENROUTER_EXPLICIT_CACHE", "0")
    assert openrouter._cacheable(messages) == messages


# Found on the extra SpokenWOZ calls -----------------------------------------------------------------------------

@pytest.mark.parametrize("turns,current,expected", [
    # MUL3774: the agent asks for the surname alone after the first name.
    ([("B", "no problem. may you please call your name."), ("A", "my name is harold."),
      ("B", "thank you, your surname."), ("A", "my name is gibson spelled g i b s o n.")], "harold", "harold gibson"),
    ([("B", "and your last name, please?"), ("A", "okay, it's moreno.")], "lucia", "lucia moreno"),
    ([("B", "and your last name, please?"), ("A", "yes.")], "lucia", None),          # no name in the answer
    ([("B", "and your last name, please?"), ("A", "moreno.")], "lucia moreno", None),  # already complete
    ([("B", "what's your surname?"), ("A", "sorry, can you repeat that")], "lucia", None),
])
def test_surname_answer_joins_first_name(turns, current, expected):
    assert spell(turns, current) == expected


def test_spells_is_not_a_surname():
    tracker = SpelledName()
    tracker.observe("A", "my name is herald spells h a r o l d.", None)
    assert tracker.take_stated() is None
    assert tracker.first_name == "herald"


@pytest.mark.parametrize("value", ["okay", "herald spells", "yes"])
def test_filler_is_not_a_name(form, value):
    quote = f"okay, my name is {value}"
    delta = parse_delta(form, {}, op("profile_name", value, quote), evidence_sources=[f"A: {quote} h a r o l d."])
    assert "profile_name" not in delta


@pytest.mark.parametrize("text,expected", [
    ("uh, yes, i'm also looking for a train and i want to go to cambridge.", {"train"}),
    ("hi, good. i am looking for a particular hotel by the name of aircon guest.", {"hotel"}),
    ("mm. i'm looking for a trend.", {"train"}),
    ("i also need a cab to the station.", {"taxi"}),
    ("i'm looking for a train and a hotel.", {"train", "hotel"}),
    ("i'm looking for a train station near the hotel.", set()),
])
def test_more_service_phrasings(form, text, expected):
    assert requested_services(form, turn(text), caller="A") == expected


@pytest.mark.parametrize("text,expected", [
    ("my name is called judith white.", None),        # "called judith" is not a name
    ("okay, my name is gracious. grace spells like g r a c e.", None),
    ("hello, my name is mary edwards.", "mary edwards"),
])
def test_stated_full_name_needs_one_sentence(text, expected):
    tracker = SpelledName()
    tracker.observe("A", text, None)
    assert tracker.take_stated() == expected


def test_agent_i_got_that_keeps_dictation_open():
    # SNG0853: "yes, i got that." between groups of digits.
    _, shown = _dictate([
        ("B", "okay. can i have your phone number."), ("A", "yes, it's 1 6 1."), ("B", "yes, i got that."),
        ("A", "3, 7, 6."), ("B", "yes, i got that."), ("A", "1 7 2 8."),
    ])
    assert shown == {"profile_phone": "1613761728"}


def test_booking_form_replies_without_facts(form):
    from strategies.delta import RESPONSE_FORMAT_SLOTS, make_messages, response_format
    messages = make_messages(form, {}, "A: i need a taxi.")
    assert '"facts"' not in messages[0]["content"].split("\n\n")[0]
    assert response_format(messages) is RESPONSE_FORMAT_SLOTS
    reply = json.dumps({"set": [{"field": "services", "value": ["taxi"], "quote": "i need a taxi"}]})
    assert parse_delta(form, {}, reply, evidence_sources=["A: i need a taxi."]) == {"services": ["taxi"]}


def test_meeting_form_keeps_facts():
    from strategies.delta import RESPONSE_FORMAT, make_messages, response_format
    data = load_dataset(load_config(), "ami", official_sample=True)
    meeting = next(iter(data.forms.values()))
    assert response_format(make_messages(meeting, {}, "A: we should use rubber.")) is RESPONSE_FORMAT
