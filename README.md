# tracecheck

Check an AI agent's recorded tool calls against the rules a healthcare workflow is supposed to follow, and say which step broke which rule.

This is an independent prototype I wrote as a student. It is not affiliated with, endorsed by, or reviewed by Wedge. Every trace in it is synthetic and comes from my own simulator. I have not seen Wedge's code, traces, or tools.

## The problem

Most agent evals I have seen look at the end state: was the appointment booked, is the referral closed. In a regulated workflow the path matters as much as the result. An agent that books the right appointment without checking coverage, or texts the wrong date, or sends appointment details before verifying who it is talking to, can leave an end state that looks correct. I wanted a check that reads the trace itself.

## Why I built it with Wedge in mind

What I could verify from public pages (checked November 2026):

- Wedge's intern posting lists "agent orchestration, prompt pipelines, and the evaluation infrastructure that keeps agents reliable inside hospitals" as the work (YC job listing).
- Their referral product page describes five stages: submit and coordinate, prior authorization "when required", outreach and scheduling, getting the consult note, and returning documentation to close the loop (wedge.health/autonomous-referral-management). My example rules follow those stages.
- Their product pages list "audit trails and source citations" and say the front office agent can "route or escalate work that needs staff involvement" (wedge.health/autonomous-front-office).
- Their YC page says engineers are forward-deployed into each health institution and that the agents are monitored and maintained afterwards (ycombinator.com/companies/wedge).

What I am assuming: I found no public Wedge code or docs, so I do not know how their agents log tool calls or what their tools are called. The tool names, the 13 rules, the 72 hour coverage window, and the timings are my guesses at a referral workflow. The one real interface I matched is the public chat-completions message format for tool calls, because it is a common way agent logs are stored. I do not know whether Wedge uses it.

## What it does

- Reads a trace, either as a list of steps or as a chat-completions style message log (`assistant` messages with `tool_calls`, answered by `tool` messages).
- Checks it against a JSON rule file and prints each violation with the step index and the rule.
- Exits with status 1 if anything was violated, so it can gate a CI run over recorded traces.

There are five rule types:

- `precedes`: every step matching `then` needs an earlier step matching `need`. Optional `if_seen` gates the rule. Optional `within` (seconds) makes it a time window.
- `responds`: every step matching `when` needs a later step matching `expect`, optionally `within` a window.
- `never`: no step may match.
- `at_most`: at most `n` matching steps, with an optional `reset`.
- `grounded`: every substring of a text field that matches a regex must appear in the result of an earlier matching step. This is how a rule says "any time quoted to the patient has to be the time in their current booking".

A value starting with `$` in a pattern is a variable. It binds on first use and must be equal everywhere else in the rule, which is how a rule says "for the same patient" when several referrals share one trace.

## Architecture

| file | role |
|---|---|
| `tracecheck.py` | matcher, the five rule types, rule validation, CLI |
| `adapters.py` | chat message log to steps and back, with a list of log problems |
| `rules_referral.json` | 13 example rules for a referral workflow |
| `sim.py` | synthetic development traces with 16 injectable faults, and the final-state baseline |
| `heldout.py` | synthetic held-out traces with 10 different faults |
| `bench.py` | runs both sets, writes `bench_results.json` |
| `demo.py` | short demo |
| `test_tracecheck.py` | 46 tests |
| `example_trace.json`, `example_chat_log.json` | inputs for the commands below |

## Key decisions and tradeoffs

**Evidence has to be visible when the call is issued.** A chat log can hold several tool calls in one assistant message. If the coverage check and the booking are issued in the same turn, the check comes first in the list but the model could not have read its result before booking. The adapter records a `turn` for every call, and `precedes` only accepts evidence from an earlier turn. A flat list of steps hides this case.

**Missing data fails closed.** A call with no result in the log has no `result` key, so it can never satisfy a `need`. A step with no timestamp can never satisfy a timed rule. The cost is real: when I dropped 2% of tool results from clean held-out logs, 613 of 2,000 traces were flagged. All 613 had a lost result, and none of the 937 traces without a lost result were flagged. I think "cannot show it was safe" is the right default for this domain, but a lossy logging pipeline would make this checker noisy.

**The adapter never raises on a bad log.** Arguments are a JSON string written by the model and can be truncated. Results are paired by call id, not position. Truncated arguments, orphan results, repeated ids, and unreadable timestamps come back as a list of problems next to whatever steps could be recovered.

**Grounding is substring matching, not understanding.** The `grounded` rule extracts timestamps with a regex and looks for the same string in the latest booking result. It is cheap and has no model in the loop, but it only sees times written in one format. "Next Tuesday morning" passes unchecked.

**The freshness rule only fires when there was a check.** `eligibility_fresh` is gated on having seen some coverage check, so a booking with no check at all gets one violation (`eligibility_before_booking`), not two.

**What I tried that did not work.** My first version had no variables. That version (`check(trace, rules, bind=False)`) flagged 1,710 of 5,000 clean development traces, because two bookings for two referrals looked like a double booking, and it caught none of the wrong-patient, wrong-specialty or unconfirmed-slot faults.

## Install and usage

Python 3, standard library only. Nothing to install. I ran it on Python 3.13.

```
python demo.py
python tracecheck.py rules_referral.json example_chat_log.json
python tracecheck.py rules_referral.json example_trace.json
python test_tracecheck.py
python bench.py
```

From Python:

```python
import adapters, tracecheck
rules = tracecheck.load_rules("rules_referral.json")
steps, problems = adapters.from_chat(messages)
for v in tracecheck.check(steps, rules):
    print(v.step, v.rule, v.message)
```

## A real example

`example_chat_log.json` is a hand-built synthetic log in which the agent issues the coverage check and the booking as parallel calls, texts a date one day off, loses the result of one write, and has one truncated argument string. `python tracecheck.py rules_referral.json example_chat_log.json` prints:

```
log problem  bad_arguments  add_note (call_0009): arguments are not a JSON object
log problem  no_result  update_referral (call_0008): no tool message answered this call
step 6  book_appointment  booked without an active eligibility check for this patient (patient=P310)  [eligibility_before_booking]
step 7  send_message  message quotes a time that is not in the patient's current booking (patient=P310): '2027-03-19 10:00'  [message_time_grounded]
step 8  update_referral  a call that changes something has no recorded result, so its outcome is unknown  [write_result_recorded]
10 steps, 13 rules, 3 violations, 2 log problems
```

A final-state check passes this trace: the right patient has the right appointment and the referral says scheduled.

## Tests and verified results

`python test_tracecheck.py` runs 46 tests (17 from the first version plus 29 new ones for time windows, turns, grounding, the adapter's failure cases, the held-out generator, and the CLI). They also pass under `python -W error -m unittest test_tracecheck`. I changed one old test by adding timestamps to its steps, because the rule file now has a timed rule and steps without timestamps fail it by design.

`python bench.py` compares the rules with a final-state check that knows the correct patient, specialty and outcome for every referral. All numbers below are from one run of that command on synthetic data.

**Development set** (seed 7, one or two referrals per trace). I wrote the rules while looking at these 16 faults, so 100% rows here are expected and carry little information.

| | trace rules | final-state check |
|---|---|---|
| 8,000 faulty traces, 16 fault types | 93.8% | 35.5% |
| false positives on 5,000 clean traces | 0 | 0 |

The one fault the rules miss entirely is a wrong weekday written in words (0 of 500).

**Held-out set** (seed 2027, three referrals per trace, written out as chat logs with parallel turns and shuffled results, then read back through the adapter). I wrote these 10 faults after freezing the rule file and did not edit the rules afterwards. `bench_results.json` records the rule file's SHA-256.

| held-out fault | trace rules | final-state check |
|---|---|---|
| booked with coverage inactive | 100.0% | 0.0% |
| authorization approved for a different referral | 100.0% | 0.0% |
| appointment details sent to another patient | 100.0% | 0.0% |
| old time quoted after a reschedule | 100.0% | 0.0% |
| failed booking confirmed to the patient | 100.0% | 100.0% |
| documents received for a different referral | 100.0% | 0.0% |
| identity verified for a different patient | 100.0% | 0.0% |
| booked after escalating to staff | 11.3% | 100.0% |
| appointment cancelled, referral left as scheduled | 0.0% | 100.0% |
| authorization submitted twice | 0.0% | 0.0% |
| all 3,000 faulty traces | 71.1% | 30.0% |
| false positives on 2,000 clean traces | 0 | 0 |

How I read this:

- 71.1% is the more honest number, and it is still generous. The same person wrote the rules and the held-out faults, and seven of the ten are "right action, wrong entity or wrong order" faults, which is exactly what variable binding is good at.
- The two checks fail on different things. Rules miss faults about state that should have changed and did not. The final-state check misses anything that leaves the end state looking right. Either one flagged 90.0% of the held-out faults, so I would run both.
- Nobody catches the duplicate authorization submission. No rule mentions it and it leaves no mark on the end state.
- The first held-out run flagged 4 of 500 clean traces. The cause was a bug in my held-out generator, which batched calls hours apart into one parallel turn and collapsed their timestamps. I fixed the generator and left the rules alone.

Speed on the development set was 945 traces per second in the run that wrote `bench_results.json`, and 1,508 and 7,681 in two earlier runs on the same shared machine, so treat it as a rough range.

## Limitations and next steps

- All traces are synthetic and both generators are mine. I have not run this on traces from a real agent or a real EHR.
- The rules are my guess at a referral workflow from public descriptions. Real rules would have to come from the people who run the workflow, and the 72 hour window is a placeholder.
- It only reads structured arguments and results, plus timestamps in one text format. It does not read what was said on a voice call.
- The chat format has no timestamps, so time rules depend on a `ts` key that I added to each message.
- Rules cannot express "state X should hold at the end", which is why the two cancellation and escalation faults get through. A next step is a small state rule type so one file covers both kinds of check.
- Each rule rescans the trace for every matching step. That is fine for traces of tens of steps and would need an index for long ones.
- Other next steps: read FHIR resources (ServiceRequest, Appointment) as evidence instead of my invented tool results, and take rules and traces from someone other than me.
