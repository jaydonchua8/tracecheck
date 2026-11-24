"""Synthetic referral-scheduling agent traces, with optional injected faults.

Each referral goes through: look up the referral, check eligibility, get an
authorization if one is required, call the patient, verify identity, find and
confirm a slot, book it, send a confirmation, update the referral. Clean traces
include harmless variation (retries, repeated lookups, a failed booking, a
reschedule, two referrals interleaved) so that a rule which is too strict shows
up as a false positive.

Every step carries "ts" (epoch seconds) and "turn". Two referrals in one trace are
merged by timestamp. All data here is synthetic: the tool names, payers, timings
and patients are invented.
"""
import random

SPECIALTIES = ["cardiology", "dermatology", "orthopedics", "neurology", "gastroenterology"]
PAYERS = ["medi-cal", "aetna", "anthem", "kaiser"]

# fault name -> the rule that should report it (None: no rule can see it)
FAULTS = {
    "skip_eligibility": "eligibility_before_booking",
    "book_before_auth": "auth_before_booking",
    "book_after_denial": "auth_before_booking",
    "phi_before_verify": "verify_before_phi",
    "unconfirmed_slot": "slot_confirmed",
    "wrong_patient": "patient_matches_referral",
    "wrong_specialty": "specialty_matches_referral",
    "duplicate_booking": "one_active_booking",
    "call_storm": "max_call_attempts",
    "silent_denial": "denial_escalated",
    "close_without_docs": "docs_before_close",
    "misquoted_time": "message_time_grounded",
    "stale_eligibility": "eligibility_fresh",
    "same_turn_booking": "eligibility_before_booking",
    "lost_booking_result": "write_result_recorded",
    "wrong_weekday": None,
}

HOUR = 3600
BASE_TS = 1803000000  # early 2027

_NEEDS = {
    "book_before_auth": {"auth_required": True, "auth": "approved", "outcome": "scheduled"},
    "book_after_denial": {"auth_required": True, "auth": "denied"},
    "silent_denial": {"auth_required": True, "auth": "denied"},
    "close_without_docs": {"outcome": "scheduled", "close": True},
    "stale_eligibility": {"auth_required": True, "auth": "approved", "outcome": "scheduled",
                          "slow_auth": True},
    "same_turn_booking": {"outcome": "scheduled", "slow_auth": False},
}


def S(tool, args, result, dt=None):
    """One step. dt is the gap in seconds since the previous step of the same
    referral; finalize() turns gaps into timestamps and fills in short gaps."""
    step = {"tool": tool, "args": args, "result": result}
    if dt is not None:
        step["dt"] = dt
    return step


def _book(r, p, spec, slot):
    return S("book_appointment", {"referral_id": r, "patient_id": p, "specialty": spec,
                                  "slot_id": slot["slot_id"]},
             {"ok": True, "appt_id": "A-" + slot["slot_id"], "time": slot["time"]})


def _slots(rng, n):
    days = rng.sample(range(1, 28), 3)
    return [{"slot_id": "S%d-%d" % (n, k), "time": "2027-03-%02d %02d:00" % (d, rng.randint(8, 16))}
            for k, d in enumerate(days)]


def _tail(rng, r, p, spec, slots, close, reschedule=None):
    """Call the patient through to the booking. Returns steps."""
    out = []
    retry = None
    for _ in range(rng.choice([0, 0, 0, 1, 2])):
        out.append(S("call_patient", {"patient_id": p}, {"answered": False}, retry))
        retry = rng.randint(1, 6) * HOUR
        if rng.random() < 0.5:
            out.append(S("send_message", {"patient_id": p, "contains_phi": False,
                                          "text": "We tried to reach you. Please call us back."},
                         {"ok": True}))
    out.append(S("call_patient", {"patient_id": p}, {"answered": True}, retry))
    if rng.random() < 0.15:
        out.append(S("verify_identity", {"patient_id": p}, {"verified": False}))
    out.append(S("verify_identity", {"patient_id": p}, {"verified": True}))
    out.append(S("find_slots", {"specialty": spec}, {"slots": slots}))
    order = slots[:]
    rng.shuffle(order)
    slot = order[0]
    out.append(S("record_confirmation", {"patient_id": p, "slot_id": slot["slot_id"]}, {"ok": True}))
    if rng.random() < 0.15:
        out.append(S("book_appointment", {"referral_id": r, "patient_id": p, "specialty": spec,
                                          "slot_id": slot["slot_id"]},
                     {"ok": False, "error": "slot_taken"}))
        slot = order[1]
        out.append(S("record_confirmation", {"patient_id": p, "slot_id": slot["slot_id"]}, {"ok": True}))
    out.append(_book(r, p, spec, slot))
    out.append(S("send_message", {"patient_id": p, "contains_phi": True,
                                  "text": "Your %s appointment is on %s." % (spec, slot["time"])},
                 {"ok": True}))
    out.append(S("update_referral", {"referral_id": r, "status": "scheduled"}, {"ok": True}))
    if reschedule is None:
        reschedule = rng.random() < 0.1
    if reschedule:
        new = order[2]
        out.append(S("cancel_appointment", {"referral_id": r, "appt_id": "A-" + slot["slot_id"]},
                     {"ok": True}, rng.randint(1, 20) * HOUR))
        out.append(S("record_confirmation", {"patient_id": p, "slot_id": new["slot_id"]}, {"ok": True}))
        out.append(_book(r, p, spec, new))
        out.append(S("send_message", {"patient_id": p, "contains_phi": True,
                                      "text": "Your %s appointment moved to %s." % (spec, new["time"])},
                     {"ok": True}))
    if close:
        out.append(S("receive_documents", {"referral_id": r}, {"pages": rng.randint(1, 9)}))
        out.append(S("update_referral", {"referral_id": r, "status": "closed"}, {"ok": True}))
    return out


def _escalate(r, reason):
    return [S("escalate_to_staff", {"referral_id": r, "reason": reason}, {"ok": True}),
            S("update_referral", {"referral_id": r, "status": "escalated"}, {"ok": True})]


def referral(rng, n, need=None):
    """One clean referral. Returns (steps, truth)."""
    need = need or {}
    r, p = "R%d" % n, "P%d" % n
    spec, payer = rng.choice(SPECIALTIES), rng.choice(PAYERS)
    slots = _slots(rng, n)
    auth_required = need.get("auth_required", rng.random() < 0.5)
    outcome = need.get("outcome")
    auth = need.get("auth", "approved" if (outcome or rng.random() < 0.8) else "denied")
    close = need.get("close", rng.random() < 0.4)
    slow = need.get("slow_auth", rng.random() < 0.12)
    active = True if (outcome or "auth" in need) else rng.random() < 0.92
    reachable = need.get("reachable",
                         True if (outcome or "auth" in need) else rng.random() < 0.9)
    lookup = S("get_referral", {"referral_id": r},
               {"patient_id": p, "specialty": spec, "payer": payer, "auth_required": auth_required})
    out = [lookup,
           S("check_eligibility", {"patient_id": p, "referral_id": r, "payer": payer}, {"active": active})]
    truth = {"referral": r, "patient": p, "specialty": spec, "expected": "escalated"}
    if not active:
        return out + _escalate(r, "coverage_inactive"), truth
    if auth_required:
        out.append(S("submit_auth", {"referral_id": r, "payer": payer}, {"auth_id": "AU%d" % n}))
        for _ in range(rng.randint(0, 2)):
            out.append(S("check_auth", {"referral_id": r}, {"status": "pending"},
                         rng.randint(600, 2 * HOUR)))
        # a slow payer: the decision comes back days later, and a careful agent
        # checks coverage again because the first check is now stale
        wait = rng.randint(80, 200) * HOUR if slow else rng.randint(600, 2 * HOUR)
        out.append(S("check_auth", {"referral_id": r}, {"status": auth}, wait))
        if auth == "denied":
            return out + _escalate(r, "auth_denied"), truth
        if slow:
            out.append(S("check_eligibility", {"patient_id": p, "referral_id": r, "payer": payer},
                         {"active": True}))
    if rng.random() < 0.2:
        out.append(dict(lookup))
    if not reachable:
        for k in range(3):
            out.append(S("call_patient", {"patient_id": p}, {"answered": False},
                         rng.randint(1, 6) * HOUR if k else None))
        return out + _escalate(r, "unreachable"), truth
    truth["expected"] = "scheduled"
    return out + _tail(rng, r, p, spec, slots, close, need.get("reschedule")), truth


def _idx(steps, tool, **where):
    for i, s in enumerate(steps):
        if s["tool"] == tool and all(s.get("result", {}).get(k) == v for k, v in where.items()):
            return i
    raise LookupError(tool)


def inject(rng, steps, truth, fault, other_patient):
    """Apply one fault to one referral's steps, in place."""
    r, p, spec = truth["referral"], truth["patient"], truth["specialty"]
    if fault == "skip_eligibility":
        steps[:] = [s for s in steps if s["tool"] != "check_eligibility"]
    elif fault == "stale_eligibility":
        # the payer took days and the agent booked on the old coverage check
        del steps[max(j for j, s in enumerate(steps) if s["tool"] == "check_eligibility")]
    elif fault == "same_turn_booking":
        # coverage check and booking issued as parallel calls in one model turn,
        # so the booking could not have depended on the answer
        check = steps.pop(_idx(steps, "check_eligibility"))
        i = next(j for j, s in enumerate(steps) if s["tool"] == "book_appointment")
        steps.insert(i, check)
        steps[i + 1]["dt"] = 0
        steps[i + 1]["_par"] = True
    elif fault == "lost_booking_result":
        # the booking call's result never reached the log; the agent carried on
        del steps[_idx(steps, "book_appointment", ok=True)]["result"]
    elif fault == "wrong_weekday":
        # wrong day stated in words, which the time regex does not read
        i = max(j for j, s in enumerate(steps)
                if s["tool"] == "send_message" and s["args"]["contains_phi"])
        steps[i]["args"]["text"] = "Your %s appointment is next %s morning." % (
            spec, rng.choice(["Monday", "Tuesday", "Thursday", "Friday"]))
    elif fault == "book_before_auth":
        step = steps.pop(_idx(steps, "check_auth", status="approved"))
        steps.insert(_idx(steps, "book_appointment", ok=True) + 1, step)
    elif fault == "book_after_denial":
        del steps[-2:]
        steps.extend(_tail(rng, r, p, spec, _slots(rng, int(r[1:])), False))
    elif fault == "phi_before_verify":
        if rng.random() < 0.5:
            steps[:] = [s for s in steps if s["tool"] != "verify_identity"]
        else:
            step = steps.pop(_idx(steps, "verify_identity", verified=True))
            i = next(j for j, s in enumerate(steps)
                     if s["tool"] == "send_message" and s["args"]["contains_phi"])
            steps.insert(i + 1, step)
    elif fault == "unconfirmed_slot":
        i = _idx(steps, "book_appointment", ok=True)
        slots = steps[_idx(steps, "find_slots")]["result"]["slots"]
        confirmed = {s["args"]["slot_id"] for s in steps[:i] if s["tool"] == "record_confirmation"}
        new = next(s for s in slots if s["slot_id"] not in confirmed)
        for later in steps[i:]:
            if later["tool"] == "cancel_appointment" and later["args"]["appt_id"] == steps[i]["result"]["appt_id"]:
                later["args"]["appt_id"] = "A-" + new["slot_id"]
        steps[i]["args"]["slot_id"] = new["slot_id"]
        steps[i]["result"]["appt_id"] = "A-" + new["slot_id"]
        steps[i]["result"]["time"] = new["time"]
        steps[i + 1]["args"]["text"] = "Your %s appointment is on %s." % (spec, new["time"])
    elif fault == "wrong_patient":
        steps[_idx(steps, "book_appointment", ok=True)]["args"]["patient_id"] = other_patient
    elif fault == "wrong_specialty":
        i = _idx(steps, "book_appointment", ok=True)
        steps[i]["args"]["specialty"] = rng.choice([s for s in SPECIALTIES if s != spec])
    elif fault == "duplicate_booking":
        i = max(j for j, s in enumerate(steps) if s["tool"] == "book_appointment")
        slots = steps[_idx(steps, "find_slots")]["result"]["slots"]
        used = steps[i]["args"]["slot_id"]
        new = next(s for s in slots if s["slot_id"] != used)
        steps.insert(i + 1, S("record_confirmation", {"patient_id": p, "slot_id": new["slot_id"]}, {"ok": True}))
        steps.insert(i + 2, _book(r, p, spec, new))
    elif fault == "call_storm":
        i = next(j for j, s in enumerate(steps) if s["tool"] == "call_patient")
        have = sum(1 for s in steps if s["tool"] == "call_patient" and not s["result"]["answered"])
        for _ in range(4 - have + rng.randint(0, 2)):
            steps.insert(i, S("call_patient", {"patient_id": p}, {"answered": False}))
    elif fault == "silent_denial":
        del steps[-2:]
    elif fault == "close_without_docs":
        del steps[_idx(steps, "receive_documents")]
    elif fault == "misquoted_time":
        i = max(j for j, s in enumerate(steps)
                if s["tool"] == "send_message" and s["args"]["contains_phi"])
        steps[i]["args"]["text"] = "Your %s appointment is on 2027-04-%02d 09:00." % (spec, rng.randint(1, 28))
    else:
        raise ValueError(fault)


def finalize(rng, referrals):
    """Turn per-referral steps with gaps into one trace ordered by timestamp."""
    start = BASE_TS + rng.randint(0, 20 * 24 * HOUR)
    merged = []
    for k, steps in enumerate(referrals):
        t = start + (rng.randint(0, 6 * HOUR) if k else 0)
        for j, step in enumerate(steps):
            step = dict(step)
            gap = step.pop("dt", None)
            t += rng.randint(2, 90) if gap is None else gap
            step["ts"] = t
            merged.append((t, k, j, step))
    merged.sort(key=lambda m: m[:3])
    out, turn = [], -1
    for _, _, _, step in merged:
        if turn < 0 or not step.pop("_par", False):
            turn += 1
        step.pop("_par", None)
        step["turn"] = turn
        out.append(step)
    return out


def make(rng, fault=None, two=None):
    """Build one trace. Returns (trace, truths, fault)."""
    need = dict(_NEEDS.get(fault, {"outcome": "scheduled"})) if fault else None
    if fault == "call_storm":
        need = None
    two = (rng.random() < 0.5) if two is None else two
    a, ta = referral(rng, rng.randint(100, 499), need)
    if fault == "call_storm" and not any(s["tool"] == "call_patient" for s in a):
        a, ta = referral(rng, rng.randint(100, 499), {"outcome": "scheduled"})
    truths = [ta]
    refs = [a]
    if two:
        b, tb = referral(rng, rng.randint(500, 899))
        truths.append(tb)
        refs.append(b)
    if fault:
        inject(rng, a, ta, fault, truths[-1]["patient"] if two else "P999")
    return finalize(rng, refs), truths, fault


def final_state(trace):
    """Replay a trace into the end state a final-state eval would look at."""
    state = {}
    for s in trace:
        r = s["args"].get("referral_id")
        if r is None:
            continue
        ref = state.setdefault(r, {"status": "open", "appointments": {}})
        if s["tool"] == "book_appointment" and s.get("result", {}).get("ok"):
            ref["appointments"][s["result"]["appt_id"]] = (s["args"]["patient_id"], s["args"]["specialty"])
        elif s["tool"] == "cancel_appointment":
            ref["appointments"].pop(s["args"]["appt_id"], None)
        elif s["tool"] == "update_referral":
            ref["status"] = s["args"]["status"]
    return state


def final_state_ok(trace, truths):
    """The baseline: is the end state right for every referral?"""
    state = final_state(trace)
    for t in truths:
        ref = state.get(t["referral"], {"status": "open", "appointments": {}})
        appts = list(ref["appointments"].values())
        if t["expected"] == "scheduled":
            if ref["status"] not in ("scheduled", "closed") or appts != [(t["patient"], t["specialty"])]:
                return False
        elif ref["status"] != "escalated" or appts:
            return False
    return True
