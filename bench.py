"""Benchmark on synthetic traces: a development set the rules were tuned on and
a held-out set they were not. Trace rules vs a check of the final state only."""
import hashlib
import json
import random
import time

import adapters
import heldout
import sim
import tracecheck

DEV_SEED, DEV_CLEAN, DEV_PER_FAULT = 7, 5000, 500
HELD_SEED, HELD_CLEAN, HELD_PER_FAULT, HELD_DROP = 2027, 2000, 300, 0.02
RULES_FILE = "rules_referral.json"


def pct(a, b):
    return "%5.1f%%" % (100.0 * a / b)


def dev(rules, results):
    rng = random.Random(DEV_SEED)
    clean = [sim.make(rng) for _ in range(DEV_CLEAN)]
    t0 = time.perf_counter()
    flagged = [bool(tracecheck.check(t, rules)) for t, _, _ in clean]
    dt = time.perf_counter() - t0
    steps = sum(len(t) for t, _, _ in clean)
    same = 0
    for t, _, _ in clean[:1000]:
        back, problems = adapters.from_chat(json.loads(json.dumps(adapters.to_chat(t))))
        same += (not problems and tracecheck.check(back, rules) == tracecheck.check(t, rules))
    c = results["dev_clean"] = {
        "traces": DEV_CLEAN,
        "two_referral_traces": sum(1 for _, tr, _ in clean if len(tr) == 2),
        "mean_steps": round(steps / float(DEV_CLEAN), 1),
        "max_steps": max(len(t) for t, _, _ in clean),
        "rules_false_positive": sum(flagged),
        "final_state_false_positive": sum(1 for t, tr, _ in clean if not sim.final_state_ok(t, tr)),
        "unbound_false_positive": sum(1 for t, _, _ in clean if tracecheck.check(t, rules, bind=False)),
        "chat_round_trip_identical": "%d of 1000" % same,
        "traces_per_second": int(DEV_CLEAN / dt),
        "steps_per_second": int(steps / dt),
    }
    tot = {"n": 0, "rules": 0, "named": 0, "final": 0, "unbound": 0}
    results["dev_faults"] = {}
    print("DEVELOPMENT SET (seed %d). The rules were written against these faults." % DEV_SEED)
    print("%-22s %6s %7s %7s %12s %14s" % ("fault", "traces", "rules", "named", "final-state", "unbound named"))
    for fault, expect in sim.FAULTS.items():
        row = {"n": DEV_PER_FAULT, "rules": 0, "named": 0, "final": 0, "unbound": 0}
        for _ in range(DEV_PER_FAULT):
            trace, truths, _ = sim.make(rng, fault)
            found = tracecheck.check(trace, rules)
            row["rules"] += bool(found)
            row["named"] += any(v.rule == expect for v in found)
            row["final"] += not sim.final_state_ok(trace, truths)
            row["unbound"] += any(v.rule == expect for v in tracecheck.check(trace, rules, bind=False))
        results["dev_faults"][fault] = row
        for k in tot:
            tot[k] += row[k]
        print("%-22s %6d %7s %7s %12s %14s" % (fault, row["n"], pct(row["rules"], row["n"]),
              pct(row["named"], row["n"]), pct(row["final"], row["n"]), pct(row["unbound"], row["n"])))
    results["dev_total"] = tot
    print("%-22s %6d %7s %7s %12s %14s" % ("all", tot["n"], pct(tot["rules"], tot["n"]),
          pct(tot["named"], tot["n"]), pct(tot["final"], tot["n"]), pct(tot["unbound"], tot["n"])))
    print("clean: %d traces (%d with two referrals), mean %.1f steps, max %d" % (
        c["traces"], c["two_referral_traces"], c["mean_steps"], c["max_steps"]))
    print("false positives on clean: rules %d, final-state %d, rules without binding %d" % (
        c["rules_false_positive"], c["final_state_false_positive"], c["unbound_false_positive"]))
    print("chat log round trip gives identical violations: %s" % c["chat_round_trip_identical"])
    print("speed: %d traces/s, %d steps/s against %d rules" % (
        c["traces_per_second"], c["steps_per_second"], len(rules)))


def held(rules, results):
    rng = random.Random(HELD_SEED)
    steps = parallel = fp = fs_fp = 0
    for _ in range(HELD_CLEAN):
        trace, problems, truths, messages = heldout.make(rng)
        steps += len(trace)
        parallel += sum(1 for m in messages if len(m.get("tool_calls") or []) > 1)
        fp += bool(tracecheck.check(trace, rules) or problems)
        fs_fp += not sim.final_state_ok(trace, truths)
    c = results["held_clean"] = {
        "traces": HELD_CLEAN, "referrals_per_trace": 3,
        "mean_steps": round(steps / float(HELD_CLEAN), 1),
        "parallel_turns": parallel,
        "rules_false_positive": fp, "final_state_false_positive": fs_fp,
    }
    lossy = {"traces": HELD_CLEAN, "drop_rate": HELD_DROP, "with_lost_result": 0, "flagged": 0,
             "flagged_with_lost_result": 0, "lost_results": 0, "by_rule": {}}
    for _ in range(HELD_CLEAN):
        trace, problems, truths, _ = heldout.make(rng, drop=HELD_DROP)
        found = tracecheck.check(trace, rules)
        lost = sum(1 for p in problems if p["kind"] == "no_result")
        lossy["lost_results"] += lost
        lossy["with_lost_result"] += bool(lost)
        lossy["flagged"] += bool(found)
        lossy["flagged_with_lost_result"] += bool(found and lost)
        for rule in {v.rule for v in found}:
            lossy["by_rule"][rule] = lossy["by_rule"].get(rule, 0) + 1
    results["held_lossy_clean"] = lossy

    tot = {"n": 0, "rules": 0, "final": 0, "either": 0}
    results["held_faults"] = {}
    print()
    print("HELD-OUT SET (seed %d, 3 referrals per trace, read through the chat-log adapter)." % HELD_SEED)
    print("No rule was written for these faults.")
    print("%-26s %6s %7s %12s  %s" % ("fault", "traces", "rules", "final-state", "rules that fired"))
    for fault in heldout.FAULTS:
        row = {"n": HELD_PER_FAULT, "rules": 0, "final": 0, "either": 0, "fired": {}}
        for _ in range(HELD_PER_FAULT):
            trace, _, truths, _ = heldout.make(rng, fault)
            found = tracecheck.check(trace, rules)
            fs = not sim.final_state_ok(trace, truths)
            row["rules"] += bool(found)
            row["final"] += fs
            row["either"] += bool(found) or fs
            for rule in {v.rule for v in found}:
                row["fired"][rule] = row["fired"].get(rule, 0) + 1
        results["held_faults"][fault] = row
        for k in tot:
            tot[k] += row[k]
        top = sorted(row["fired"], key=lambda r: -row["fired"][r])[:2]
        print("%-26s %6d %7s %12s  %s" % (fault, row["n"], pct(row["rules"], row["n"]),
              pct(row["final"], row["n"]), ", ".join(top) or "none"))
    results["held_total"] = tot
    print("%-26s %6d %7s %12s  either check: %s" % ("all", tot["n"], pct(tot["rules"], tot["n"]),
          pct(tot["final"], tot["n"]), pct(tot["either"], tot["n"]).strip()))
    print("clean: %d traces, mean %.1f steps, %d parallel turns; false positives: rules %d, final-state %d" % (
        c["traces"], c["mean_steps"], c["parallel_turns"], c["rules_false_positive"],
        c["final_state_false_positive"]))
    print("clean with %.0f%% of tool results dropped from the log: %d of %d traces lost a result, "
          "%d were flagged, %d flagged without a lost result" % (
              100 * HELD_DROP, lossy["with_lost_result"], lossy["traces"], lossy["flagged"],
              lossy["flagged"] - lossy["flagged_with_lost_result"]))


def main():
    rules = tracecheck.load_rules(RULES_FILE)
    with open(RULES_FILE, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    results = {"synthetic": True, "rules": len(rules), "rules_sha256": digest,
               "dev_seed": DEV_SEED, "held_seed": HELD_SEED}
    dev(rules, results)
    held(rules, results)
    print()
    print("rules file sha256 %s" % digest[:16])
    with open("bench_results.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
