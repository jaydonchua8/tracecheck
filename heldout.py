"""Held-out evaluation set. Synthetic, like sim.py, but not what the rules were tuned on.

I wrote this file after freezing rules_referral.json and did not edit the rules
afterwards (bench.py records the rule file's SHA-256). It differs from the
development set in sim.py in four ways:

  - different seed, and always three referrals merged into one trace instead of one or two
  - benign steps the development set never had: staff notes, inbound calls,
    repeated lookups, and reminder messages that quote the appointment time
  - every trace is written out as a chat-completions message log (adapters.to_chat),
    with read-only calls batched into parallel turns and tool messages shuffled,
    then read back through adapters.from_chat
  - ten fault types that are not in sim.FAULTS. No rule was written for them.

The same person wrote the rules and this file, so this is weaker than a blind
test set from someone else. It measures how the rules do on faults I did not
have in mind when writing them, not how they do on a real agent.
"""
import json

import adapters
import sim

READ_ONLY = ("get_referral", "check_eligibility", "check_auth", "find_slots")

# fault -> the referral it needs
FAULTS = {
    "inactive_coverage_booked": {"outcome": "scheduled"},
    "auth_for_other_referral": {"auth_required": True, "auth": "approved", "outcome": "scheduled"},
    "phi_to_other_patient": {"outcome": "scheduled"},
    "old_time_after_reschedule": {"outcome": "scheduled", "reschedule": True},
    "failed_booking_confirmed": {"outcome": "scheduled", "reschedule": False},
    "docs_for_other_referral": {"outcome": "scheduled", "close": True},
    "verified_other_patient": {"outcome": "scheduled"},
    "booked_after_escalation": {"reachable": False, "auth_required": False},
    "cancelled_left_scheduled": {"outcome": "scheduled", "reschedule": False, "close": False},
    "auth_submitted_twice": {"auth_required": True, "auth": "approved", "outcome": "scheduled"},
}


def _noise(rng, steps, truth):
    """Benign extra steps for one referral."""
    r, p = truth["referral"], truth["patient"]
    out = []
    for s in steps:
        out.append(s)
        if rng.random() < 0.08:
            out.append(sim.S("add_note", {"referral_id": r, "text": "Patient prefers mornings."},
                             {"ok": True}))
        if s["tool"] == "call_patient" and not s["result"]["answered"] and rng.random() < 0.3:
            out.append(sim.S("receive_call", {"patient_id": p}, {"reason": "callback"},
                             rng.randint(300, 3600)))
        if s["tool"] == "get_referral" and rng.random() < 0.15:
            out.append(sim.S("get_referral", dict(s["args"]), dict(s["result"])))
    return out


def _reminder(rng, steps, truth):
    """A reminder that quotes the current booking, sent a few hours after the last one."""
    books = [s for s in steps if s["tool"] == "book_appointment" and s.get("result", {}).get("ok")]
    if not books or truth["expected"] != "scheduled" or rng.random() > 0.5:
        return
    i = max(j for j, s in enumerate(steps) if s["tool"] == "send_message")
    steps.insert(i + 1, sim.S("send_message", {
        "patient_id": truth["patient"], "contains_phi": True,
        "text": "Reminder: your appointment is on %s." % books[-1]["result"]["time"]},
        {"ok": True}, rng.randint(1, 5) * sim.HOUR))


def inject(rng, steps, truth, fault, other):
    """Apply one held-out fault to the first referral, in place. `other` is the
    truth of another referral in the same trace."""
    r, p, spec = truth["referral"], truth["patient"], truth["specialty"]
    if fault == "inactive_coverage_booked":
        for s in steps:
            if s["tool"] == "check_eligibility":
                s["result"] = {"active": False}
    elif fault == "auth_for_other_referral":
        i = sim._idx(steps, "check_auth", status="approved")
        steps[i]["args"] = {"referral_id": other["referral"]}
    elif fault == "phi_to_other_patient":
        i = next(j for j, s in enumerate(steps)
                 if s["tool"] == "send_message" and s["args"]["contains_phi"])
        steps[i]["args"]["patient_id"] = other["patient"]
    elif fault == "old_time_after_reschedule":
        books = [s for s in steps if s["tool"] == "book_appointment" and s["result"].get("ok")]
        i = max(j for j, s in enumerate(steps) if s["tool"] == "send_message")
        steps[i]["args"]["text"] = "Your %s appointment is on %s." % (spec, books[0]["result"]["time"])
    elif fault == "failed_booking_confirmed":
        ok = sim._idx(steps, "book_appointment", ok=True)
        steps[ok]["result"] = {"ok": False, "error": "slot_taken"}
    elif fault == "docs_for_other_referral":
        steps[sim._idx(steps, "receive_documents")]["args"] = {"referral_id": other["referral"]}
    elif fault == "verified_other_patient":
        for s in steps:
            if s["tool"] == "verify_identity":
                s["args"] = {"patient_id": other["patient"]}
    elif fault == "booked_after_escalation":
        # staff were told the patient is unreachable; the agent books anyway a day later
        tail = sim._tail(rng, r, p, spec, sim._slots(rng, int(r[1:])), False, False)
        tail = [s for s in tail if not (s["tool"] == "call_patient" and not s["result"]["answered"])]
        tail[0]["dt"] = rng.randint(20, 40) * sim.HOUR
        steps.extend(tail)
    elif fault == "cancelled_left_scheduled":
        book = steps[sim._idx(steps, "book_appointment", ok=True)]
        steps.append(sim.S("cancel_appointment", {"referral_id": r,
                                                  "appt_id": book["result"]["appt_id"]},
                           {"ok": True}, rng.randint(1, 20) * sim.HOUR))
    elif fault == "auth_submitted_twice":
        i = sim._idx(steps, "submit_auth")
        steps.insert(i + 1, sim.S("submit_auth", dict(steps[i]["args"]),
                                  {"auth_id": steps[i]["result"]["auth_id"] + "b"}))
    else:
        raise ValueError(fault)


def _batch(trace):
    """Put runs of consecutive read-only calls into one parallel turn. Only calls
    issued within two minutes of the start of the turn are batched, because the
    chat format keeps one timestamp per assistant message."""
    turn = -1
    prev_read, start = False, 0
    for s in trace:
        read = s["tool"] in READ_ONLY
        if not (read and prev_read and s["ts"] - start <= 120):
            turn += 1
            start = s["ts"]
        s["turn"] = turn
        prev_read = read
    return trace


def make(rng, fault=None, drop=0.0):
    """One held-out trace. Returns (steps, problems, truths, messages).

    drop is the chance that each tool message is lost from the log, to model a
    truncated or lossy log. Steps come back through the chat adapter."""
    numbers = rng.sample(range(100, 900), 3)
    refs, truths = [], []
    for k, n in enumerate(numbers):
        need = dict(FAULTS[fault]) if (fault and k == 0) else None
        steps, truth = sim.referral(rng, n, need)
        refs.append(steps)
        truths.append(truth)
    if fault:
        inject(rng, refs[0], truths[0], fault, truths[1])
    for k in range(3):
        if not (fault and k == 0):
            _reminder(rng, refs[k], truths[k])
        refs[k] = _noise(rng, refs[k], truths[k])
    trace = _batch(sim.finalize(rng, refs))
    messages = adapters.to_chat(trace, shuffle=rng)
    if drop:
        messages = [m for m in messages if m["role"] != "tool" or rng.random() >= drop]
    messages = json.loads(json.dumps(messages))
    steps, problems = adapters.from_chat(messages)
    return steps, problems, truths, messages
