"""Demo: check a recorded chat log, then a small held-out comparison. Takes a few seconds.

All data is synthetic."""
import random

import heldout
import sim
import tracecheck


def main():
    rules = tracecheck.load_rules("rules_referral.json")

    print("1. One chat-completions style log (example_chat_log.json)")
    print("   The agent issued the coverage check and the booking as parallel calls,")
    print("   texted the wrong date, and one result and one argument string are damaged.")
    print()
    trace, problems = tracecheck.load_trace("example_chat_log.json")
    for i, s in enumerate(trace):
        print("   step %d  turn %d  %s%s" % (i, s["turn"], s["tool"],
                                           "" if "result" in s else "   (no result in log)"))
    print()
    for p in problems:
        print("   log problem  %s: %s" % (p["kind"], p["detail"]))
    for v in tracecheck.check(trace, rules):
        print("   VIOLATION step %d [%s] %s" % (v.step, v.rule, v.message))

    print()
    print("2. 100 held-out traces per fault (3 referrals each, no rule written for these faults)")
    print("   %-26s %6s %12s" % ("fault", "rules", "final-state"))
    rng = random.Random(5)
    for fault in heldout.FAULTS:
        hit = final = 0
        for _ in range(100):
            steps, _, truths, _ = heldout.make(rng, fault)
            hit += bool(tracecheck.check(steps, rules))
            final += not sim.final_state_ok(steps, truths)
        print("   %-26s %5d%% %11d%%" % (fault, hit, final))
    flagged = sum(bool(tracecheck.check(heldout.make(rng)[0], rules)) for _ in range(300))
    print("   clean held-out traces flagged: %d of 300" % flagged)


if __name__ == "__main__":
    main()
