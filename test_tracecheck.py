import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

import adapters
import heldout
import sim
import tracecheck
from tracecheck import check, match, validate

RULES = tracecheck.load_rules("rules_referral.json")


def S(tool, args=None, result=None):
    return {"tool": tool, "args": args or {}, "result": result or {}}


def ids(trace, rules=RULES, **kw):
    return [(v.rule, v.step) for v in check(trace, rules, **kw)]


class MatchTest(unittest.TestCase):
    def test_literal_and_tool(self):
        step = S("a", {"x": 1})
        self.assertEqual(match({"tool": "a", "where": {"args.x": 1}}, step), {})
        self.assertIsNone(match({"tool": "b"}, step))
        self.assertIsNone(match({"tool": "a", "where": {"args.x": 2}}, step))
        self.assertEqual(match({"tool": ["a", "b"]}, step), {})

    def test_true_is_not_one(self):
        self.assertIsNone(match({"where": {"result.ok": True}}, S("a", result={"ok": 1})))

    def test_variable_binds_then_must_agree(self):
        pat = {"where": {"args.p": "$p"}}
        env = match(pat, S("a", {"p": "P1"}))
        self.assertEqual(env, {"$p": "P1"})
        self.assertIsNone(match(pat, S("a", {"p": "P2"}), env))
        self.assertIsNone(match(pat, S("a", {})))

    def test_operators(self):
        step = S("a", {"n": 3})
        self.assertIsNotNone(match({"where": {"args.n": {"gte": 3, "lte": 3}}}, step))
        self.assertIsNotNone(match({"where": {"args.n": {"in": [1, 3]}}}, step))
        self.assertIsNone(match({"where": {"args.n": {"ne": 3}}}, step))
        self.assertIsNotNone(match({"where": {"args.zz": {"exists": False}}}, step))
        self.assertIsNone(match({"where": {"args.zz": {"ne": 1}}}, step))


class RuleTest(unittest.TestCase):
    def test_precedes_is_per_binding(self):
        trace = [S("check_eligibility", {"patient_id": "P1"}, {"active": True}),
                 S("record_confirmation", {"patient_id": "P2", "slot_id": "s"}),
                 S("book_appointment", {"patient_id": "P2", "slot_id": "s"}, {"ok": True})]
        for n, step in enumerate(trace):  # eligibility_fresh is timed, so steps need a time
            step["ts"] = 1000 + n
        self.assertEqual(ids(trace), [("eligibility_before_booking", 2)])
        self.assertEqual(ids(trace, bind=False), [])

    def test_need_must_come_first(self):
        trace = [S("send_message", {"patient_id": "P1", "contains_phi": True}),
                 S("verify_identity", {"patient_id": "P1"}, {"verified": True})]
        self.assertEqual(ids(trace), [("verify_before_phi", 0)])
        self.assertEqual(ids(trace[::-1]), [])

    def test_failed_verification_does_not_count(self):
        trace = [S("verify_identity", {"patient_id": "P1"}, {"verified": False}),
                 S("send_message", {"patient_id": "P1", "contains_phi": True})]
        self.assertEqual(ids(trace), [("verify_before_phi", 1)])

    def test_non_phi_message_is_fine(self):
        self.assertEqual(ids([S("send_message", {"patient_id": "P1", "contains_phi": False})]), [])

    def test_if_seen_gates_the_rule(self):
        rule = [r for r in RULES if r["id"] == "auth_before_booking"]
        book = S("book_appointment", {"referral_id": "R1"}, {"ok": True})
        no_auth = S("get_referral", {"referral_id": "R1"}, {"auth_required": False})
        needs = S("get_referral", {"referral_id": "R1"}, {"auth_required": True})
        other = S("check_auth", {"referral_id": "R2"}, {"status": "approved"})
        mine = S("check_auth", {"referral_id": "R1"}, {"status": "approved"})
        self.assertEqual(ids([no_auth, book], rule), [])
        self.assertEqual(ids([needs, other, book], rule), [("auth_before_booking", 2)])
        self.assertEqual(ids([needs, mine, book], rule), [])
        self.assertEqual(ids([needs, book, mine], rule), [("auth_before_booking", 1)])

    def test_responds(self):
        denied = S("check_auth", {"referral_id": "R1"}, {"status": "denied"})
        self.assertEqual(ids([denied]), [("denial_escalated", 0)])
        self.assertEqual(ids([denied, S("escalate_to_staff", {"referral_id": "R2"})]),
                         [("denial_escalated", 0)])
        self.assertEqual(ids([denied, S("escalate_to_staff", {"referral_id": "R1"})]), [])

    def test_at_most_counts_per_binding_and_resets(self):
        rule = [r for r in RULES if r["id"] == "one_active_booking"]
        b1 = S("book_appointment", {"referral_id": "R1"}, {"ok": True})
        b2 = S("book_appointment", {"referral_id": "R2"}, {"ok": True})
        failed = S("book_appointment", {"referral_id": "R1"}, {"ok": False})
        cancel = S("cancel_appointment", {"referral_id": "R1"})
        self.assertEqual(ids([b1, b2], rule), [])
        self.assertEqual(ids([failed, b1], rule), [])
        self.assertEqual(ids([b1, cancel, b1], rule), [])
        self.assertEqual(ids([b1, b2, b1, b1], rule), [("one_active_booking", 2)])
        self.assertEqual(ids([b1, b2], rule, bind=False), [("one_active_booking", 1)])

    def test_never_with_after(self):
        rule = [{"id": "x", "type": "never", "match": {"tool": "b", "where": {"args.k": "$k"}},
                 "after": {"tool": "a", "where": {"args.k": "$k"}}}]
        self.assertEqual(ids([S("b", {"k": 1}), S("a", {"k": 1})], rule), [])
        self.assertEqual(ids([S("a", {"k": 2}), S("b", {"k": 1})], rule), [])
        self.assertEqual(ids([S("a", {"k": 1}), S("b", {"k": 1})], rule), [("x", 1)])

    def test_validate_rejects_bad_rules(self):
        for bad in ([{"id": "a", "type": "sometimes"}],
                    [{"id": "a", "type": "precedes", "then": {}}],
                    [{"id": "a", "type": "never", "match": {}}, {"id": "a", "type": "never", "match": {}}],
                    [{"id": "a", "type": "at_most", "n": 1, "match": {"where": {"args.r": "$r"}},
                      "reset": {"tool": "c"}}]):
            with self.assertRaises(ValueError):
                validate(bad)


class SimTest(unittest.TestCase):
    def test_clean_traces_have_no_violations(self):
        rng = random.Random(11)
        for _ in range(400):
            trace, truths, _ = sim.make(rng)
            self.assertEqual(check(trace, RULES), [])
            self.assertTrue(sim.final_state_ok(trace, truths))

    def test_each_fault_is_reported_by_its_rule(self):
        rng = random.Random(12)
        for fault, expect in sim.FAULTS.items():
            for two in (False, True):
                for _ in range(40):
                    trace, _, _ = sim.make(rng, fault, two=two)
                    found = {v.rule for v in check(trace, RULES)}
                    if expect is None:
                        self.assertEqual(found, set(), fault)
                    else:
                        self.assertIn(expect, found, fault)

    def test_violation_points_at_the_booking_step(self):
        rng = random.Random(13)
        trace, _, _ = sim.make(rng, "unconfirmed_slot", two=True)
        v = [x for x in check(trace, RULES) if x.rule == "slot_confirmed"][0]
        self.assertEqual(trace[v.step]["tool"], "book_appointment")
        self.assertIn("slot=", v.message)

    def test_trace_survives_json_round_trip(self):
        trace, _, _ = sim.make(random.Random(14), "wrong_patient")
        again = json.loads(json.dumps(trace))
        self.assertEqual(check(again, RULES), check(trace, RULES))


def only(rule_id):
    return [r for r in RULES if r["id"] == rule_id]


def call(cid, name, args):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


def tool(cid, content):
    return {"role": "tool", "tool_call_id": cid,
            "content": content if isinstance(content, str) else json.dumps(content)}


class TimeAndTurnTest(unittest.TestCase):
    def setUp(self):
        self.elig = S("check_eligibility", {"patient_id": "P1"}, {"active": True})
        self.book = S("book_appointment", {"patient_id": "P1"}, {"ok": True})

    def test_within_window(self):
        rule = only("eligibility_fresh")
        hour = 3600
        for gap, expect in ((71 * hour, []), (72 * hour, []), (73 * hour, [("eligibility_fresh", 1)])):
            trace = [dict(self.elig, ts=0), dict(self.book, ts=gap)]
            self.assertEqual(ids(trace, rule), expect, gap)

    def test_a_fresh_recheck_clears_a_stale_one(self):
        trace = [dict(self.elig, ts=0), dict(self.elig, ts=500000), dict(self.book, ts=500100)]
        self.assertEqual(ids(trace, only("eligibility_fresh")), [])

    def test_missing_timestamp_fails_closed(self):
        rule = only("eligibility_fresh")
        self.assertEqual(ids([self.elig, dict(self.book, ts=5)], rule), [("eligibility_fresh", 1)])
        self.assertEqual(ids([dict(self.elig, ts=1), self.book], rule), [("eligibility_fresh", 1)])
        self.assertEqual(ids([dict(self.elig, ts="1"), dict(self.book, ts=5)], rule),
                         [("eligibility_fresh", 1)])

    def test_fresh_rule_stays_quiet_when_there_was_no_check_at_all(self):
        self.assertEqual(ids([dict(self.book, ts=5)], only("eligibility_fresh")), [])

    def test_timed_responds(self):
        rule = [{"id": "r", "type": "responds", "within": 10,
                 "when": {"tool": "a"}, "expect": {"tool": "b"}}]
        a, b = S("a"), S("b")
        self.assertEqual(ids([dict(a, ts=0), dict(b, ts=10)], rule), [])
        self.assertEqual(ids([dict(a, ts=0), dict(b, ts=11)], rule), [("r", 0)])

    def test_same_turn_result_is_not_evidence(self):
        rule = only("eligibility_before_booking")
        self.assertEqual(ids([dict(self.elig, turn=3), dict(self.book, turn=3)], rule),
                         [("eligibility_before_booking", 1)])
        self.assertEqual(ids([dict(self.elig, turn=2), dict(self.book, turn=3)], rule), [])
        self.assertEqual(ids([self.elig, dict(self.book, turn=3)], rule), [])

    def test_validate_rejects_bad_within_and_regex(self):
        ok = {"id": "a", "type": "precedes", "then": {}, "need": {}}
        for bad in ([dict(ok, within=-1)], [dict(ok, within="1h")], [dict(ok, within=True)],
                    [{"id": "a", "type": "never", "match": {}, "within": 5}],
                    [{"id": "a", "type": "at_most", "match": {}, "n": "2"}],
                    [{"id": "a", "type": "grounded", "match": {}, "field": "args.t",
                      "extract": "(", "sources": {}}],
                    [{"id": "a", "type": "grounded", "match": {}, "field": "args.t", "sources": {}}]):
            with self.assertRaises(ValueError):
                validate(bad)


class GroundedTest(unittest.TestCase):
    def setUp(self):
        self.rule = only("message_time_grounded")

    def book(self, patient, when, ok=True):
        return S("book_appointment", {"patient_id": patient}, {"ok": ok, "time": when})

    def msg(self, patient, text):
        return S("send_message", {"patient_id": patient, "text": text})

    def test_quoted_time_must_match_the_booking(self):
        b = self.book("P1", "2027-03-18 10:00")
        self.assertEqual(ids([b, self.msg("P1", "See you on 2027-03-18 10:00.")], self.rule), [])
        self.assertEqual(ids([b, self.msg("P1", "See you on 2027-03-19 10:00.")], self.rule),
                         [("message_time_grounded", 1)])
        v = check([b, self.msg("P1", "See you on 2027-03-19 10:00.")], self.rule)[0]
        self.assertIn("2027-03-19 10:00", v.message)

    def test_text_without_a_time_is_not_checked(self):
        self.assertEqual(ids([self.msg("P1", "Please call us back.")], self.rule), [])
        self.assertEqual(ids([S("send_message", {"patient_id": "P1"})], self.rule), [])
        self.assertEqual(ids([S("send_message", {"patient_id": "P1", "text": 7})], self.rule), [])

    def test_another_patients_booking_is_not_a_source(self):
        trace = [self.book("P2", "2027-03-18 10:00"), self.msg("P1", "On 2027-03-18 10:00.")]
        self.assertEqual(ids(trace, self.rule), [("message_time_grounded", 1)])

    def test_only_the_latest_booking_counts(self):
        old, new = self.book("P1", "2027-03-18 10:00"), self.book("P1", "2027-03-20 14:00")
        self.assertEqual(ids([old, new, self.msg("P1", "Moved to 2027-03-20 14:00.")], self.rule), [])
        self.assertEqual(ids([old, new, self.msg("P1", "On 2027-03-18 10:00.")], self.rule),
                         [("message_time_grounded", 2)])

    def test_failed_or_later_booking_is_not_a_source(self):
        failed = self.book("P1", "2027-03-18 10:00", ok=False)
        m = self.msg("P1", "On 2027-03-18 10:00.")
        self.assertEqual(ids([failed, m], self.rule), [("message_time_grounded", 1)])
        self.assertEqual(ids([m, self.book("P1", "2027-03-18 10:00")], self.rule),
                         [("message_time_grounded", 0)])

    def test_every_quoted_time_is_checked(self):
        trace = [self.book("P1", "2027-03-18 10:00"),
                 self.msg("P1", "Was 2027-03-11 09:00, now 2027-03-18 10:00.")]
        v = check(trace, self.rule)
        self.assertEqual(len(v), 1)
        self.assertIn("2027-03-11 09:00", v[0].message)
        self.assertNotIn("'2027-03-18 10:00'", v[0].message)


class ChatAdapterTest(unittest.TestCase):
    def test_pairs_results_by_id_not_position(self):
        log = [{"role": "system", "content": "x"},
               {"role": "assistant", "content": None, "ts": 100,
                "tool_calls": [call("c1", "a", {"k": 1}), call("c2", "b", {"k": 2})]},
               tool("c2", {"v": "two"}), tool("c1", {"v": "one"}),
               {"role": "assistant", "content": "done"},
               {"role": "assistant", "tool_calls": [call("c3", "c", {})]},
               tool("c3", {"ok": True})]
        steps, problems = adapters.from_chat(log)
        self.assertEqual(problems, [])
        self.assertEqual([(s["tool"], s["args"], s["result"], s["turn"]) for s in steps],
                         [("a", {"k": 1}, {"v": "one"}, 0), ("b", {"k": 2}, {"v": "two"}, 0),
                          ("c", {}, {"ok": True}, 1)])
        self.assertEqual(steps[0]["ts"], 100)
        self.assertNotIn("ts", steps[2])

    def test_missing_result_leaves_no_result_key(self):
        steps, problems = adapters.from_chat(
            [{"role": "assistant", "tool_calls": [call("c1", "book_appointment", {"patient_id": "P1"})]}])
        self.assertNotIn("result", steps[0])
        self.assertEqual([p["kind"] for p in problems], ["no_result"])
        self.assertEqual(ids(steps, only("write_result_recorded")), [("write_result_recorded", 0)])

    def test_malformed_arguments_are_kept_and_reported(self):
        for raw in ('{"patient_id": "P1', "[1, 2]", "null", ""):
            steps, problems = adapters.from_chat(
                [{"role": "assistant", "tool_calls": [call("c1", "a", raw)]}, tool("c1", {"ok": True})])
            self.assertEqual(steps[0]["args"], {}, raw)
            self.assertEqual(steps[0]["args_raw"], raw)
            self.assertEqual([p["kind"] for p in problems], ["bad_arguments"], raw)

    def test_orphan_duplicate_and_garbage_messages(self):
        log = ["not a message", {"content": "no role"},
               {"role": "assistant", "tool_calls": [call("c1", "a", {}), call("c1", "b", {}),
                                                    {"id": "c9"}, "junk"]},
               tool("c1", {"n": 1}), tool("c1", {"n": 2}), tool("zzz", {"n": 3}),
               {"role": "assistant", "tool_calls": "oops"}]
        steps, problems = adapters.from_chat(log)
        kinds = sorted(p["kind"] for p in problems)
        self.assertEqual(kinds, ["bad_tool_call", "bad_tool_call", "bad_tool_call", "duplicate_call_id",
                                 "duplicate_result", "no_result", "not_a_message", "not_a_message",
                                 "orphan_result"])
        self.assertEqual([s["tool"] for s in steps], ["a", "b"])
        self.assertEqual(steps[0]["result"], {"n": 1})
        self.assertNotIn("result", steps[1])

    def test_non_json_and_non_object_content(self):
        steps, _ = adapters.from_chat(
            [{"role": "assistant", "tool_calls": [call("c1", "a", {}), call("c2", "b", {})]},
             tool("c1", "Service unavailable"), tool("c2", "[1, 2]")])
        self.assertEqual(steps[0]["result"], {"text": "Service unavailable"})
        self.assertEqual(steps[1]["result"], {"value": [1, 2]})

    def test_timestamps(self):
        def one(ts):
            return adapters.from_chat([{"role": "assistant", "ts": ts, "tool_calls": [call("c", "a", {})]},
                                       tool("c", {})])
        self.assertEqual(one("2027-03-01T15:00:00Z")[0][0]["ts"], 1803913200.0)
        self.assertEqual(one("2027-03-01T10:00:00-05:00")[0][0]["ts"], 1803913200.0)
        for bad in ("yesterday", True, None, [1]):
            steps, problems = one(bad)
            self.assertNotIn("ts", steps[0])
            self.assertEqual([p["kind"] for p in problems], ["bad_timestamp"])

    def test_round_trip_keeps_violations(self):
        rng = random.Random(21)
        for fault in list(sim.FAULTS) + [None] * 5:
            trace, _, _ = sim.make(rng, fault, two=True)
            log = json.loads(json.dumps(adapters.to_chat(trace, shuffle=rng)))
            back, problems = adapters.from_chat(log)
            self.assertEqual(check(back, RULES), check(trace, RULES), fault)
            self.assertEqual(bool(problems), fault == "lost_booking_result", fault)

    def test_parallel_booking_in_a_chat_log_is_caught(self):
        log = [{"role": "assistant", "ts": 10, "tool_calls": [
                   call("c1", "check_eligibility", {"patient_id": "P1"}),
                   call("c2", "record_confirmation", {"patient_id": "P1", "slot_id": "s"})]},
               tool("c1", {"active": True}), tool("c2", {"ok": True}),
               {"role": "assistant", "ts": 20, "tool_calls": [
                   call("c3", "book_appointment", {"patient_id": "P1", "slot_id": "s"})]},
               tool("c3", {"ok": True})]
        self.assertEqual(ids(adapters.from_chat(log)[0]), [])
        log[0]["tool_calls"].append(log[3]["tool_calls"].pop())
        del log[3]
        found = ids(adapters.from_chat(log)[0])
        self.assertEqual(found, [("eligibility_before_booking", 2), ("slot_confirmed", 2)])


class HeldOutTest(unittest.TestCase):
    def test_clean_heldout_traces_have_no_violations(self):
        rng = random.Random(31)
        for _ in range(150):
            steps, problems, truths, messages = heldout.make(rng)
            self.assertEqual(problems, [])
            self.assertEqual(check(steps, RULES), [])
            self.assertTrue(sim.final_state_ok(steps, truths))
            self.assertEqual(len(truths), 3)

    def test_heldout_faults_do_not_overlap_dev_faults(self):
        self.assertEqual(set(heldout.FAULTS) & set(sim.FAULTS), set())

    def test_every_heldout_fault_changes_the_trace_and_checker_survives(self):
        for fault in heldout.FAULTS:
            a = heldout.make(random.Random(32), fault)[3]
            b = heldout.make(random.Random(32))[3]
            self.assertNotEqual(a, b, fault)
            for seed in range(20):
                steps, _, _, _ = heldout.make(random.Random(seed), fault)
                check(steps, RULES)

    def test_lost_results_never_crash_and_only_flag_lossy_traces(self):
        rng = random.Random(33)
        lossy = 0
        for _ in range(150):
            steps, problems, _, _ = heldout.make(rng, drop=0.05)
            found = check(steps, RULES)
            if found:
                self.assertTrue(any(p["kind"] == "no_result" for p in problems))
            lossy += bool(problems)
        self.assertGreater(lossy, 50)

    def test_timestamps_are_ordered(self):
        steps = heldout.make(random.Random(34))[0]
        self.assertEqual([s["ts"] for s in steps], sorted(s["ts"] for s in steps))
        self.assertEqual([s["turn"] for s in steps], sorted(s["turn"] for s in steps))


class CliTest(unittest.TestCase):
    def run_cli(self, *args):
        p = subprocess.run([sys.executable, "tracecheck.py"] + list(args),
                           capture_output=True, text=True)
        return p.returncode, p.stdout

    def test_example_files(self):
        code, out = self.run_cli("rules_referral.json", "example_trace.json")
        self.assertEqual(code, 1)
        self.assertIn("[auth_before_booking]", out)
        self.assertIn("1 violations", out)
        code, out = self.run_cli("rules_referral.json", "example_chat_log.json")
        self.assertEqual(code, 1)
        for want in ("[eligibility_before_booking]", "[message_time_grounded]",
                     "[write_result_recorded]", "bad_arguments", "no_result", "3 violations"):
            self.assertIn(want, out)

    def test_bad_input_exits_2_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as d:
            for name, text in (("a.json", "{not json"), ("b.json", '[{"no_tool": 1}]'), ("c.json", "[3]")):
                path = os.path.join(d, name)
                with open(path, "w") as f:
                    f.write(text)
                code, out = self.run_cli("rules_referral.json", path)
                self.assertEqual(code, 2, name)
                self.assertTrue(out.startswith("error:"), out)
            code, out = self.run_cli("rules_referral.json", os.path.join(d, "missing.json"))
            self.assertEqual(code, 2)
            code, out = self.run_cli(os.path.join(d, "a.json"), "example_trace.json")
            self.assertEqual(code, 2)

    def test_clean_trace_exits_0(self):
        trace, _, _ = sim.make(random.Random(41))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.jsonl")
            with open(path, "w") as f:
                f.write("\n".join(json.dumps(s) for s in trace))
            code, out = self.run_cli("rules_referral.json", path)
        self.assertEqual(code, 0, out)
        self.assertIn("0 violations", out)


if __name__ == "__main__":
    unittest.main()
