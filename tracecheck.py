"""tracecheck: check an agent's tool-call trace against ordered workflow rules.

A trace is a list of steps: {"tool": str, "args": {...}, "result": {...}}.
Optional step keys: "ts" (seconds, when the call was issued) and "turn" (which
model turn issued the call; calls issued in parallel share a turn). A step whose
result never reached the log has no "result" key at all.
A rule file is a JSON list of rules. Five rule types:

  precedes  every step matching `then` needs an earlier step matching `need`
            (only enforced when `if_seen` matched earlier, if `if_seen` is given)
  responds  every step matching `when` needs a later step matching `expect`
  never     no step may match `match` (optionally: only once `after` has happened)
  at_most   at most `n` steps may match `match` per binding; `reset` restarts the count
  grounded  every substring of the text at `field` that matches the regex `extract`
            must appear in the result of an earlier step matching `sources`
            (`latest`: only the most recent such step counts)

`precedes` and `responds` take an optional `within` (seconds). A step with no
timestamp can never satisfy a timed rule. An earlier step only counts as evidence
if its result was visible when the later call was issued, so a step issued in the
same parallel turn does not count.

A pattern is {"tool": name or [names], "where": {dotted.path: spec}}.
A spec is a literal, an operator dict ({"in": [...]}, {"ne": x}, {"exists": bool},
{"gte": x}, {"lte": x}), or a variable "$name". A variable binds on first use and
must be equal everywhere else in the same rule, which is how a rule says
"for the same patient" or "for the same referral".
"""
import json
import re
import sys
from collections import namedtuple

MISSING = object()
Violation = namedtuple("Violation", "rule step message")


def get(step, path):
    cur = step
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return MISSING
        cur = cur[part]
    return cur


def _spec_ok(spec, value, env, bind):
    """Return the (possibly extended) env if value satisfies spec, else None."""
    if isinstance(spec, str) and spec.startswith("$"):
        if value is MISSING:
            return None
        if not bind:
            return env
        if spec in env:
            return env if env[spec] == value else None
        new = dict(env)
        new[spec] = value
        return new
    if isinstance(spec, dict):
        for op, arg in spec.items():
            if op == "exists":
                ok = (value is not MISSING) == bool(arg)
            elif value is MISSING:
                ok = False
            elif op == "in":
                ok = value in arg
            elif op == "ne":
                ok = value != arg
            elif op == "eq":
                ok = value == arg
            elif op == "gte":
                ok = value >= arg
            elif op == "lte":
                ok = value <= arg
            else:
                raise ValueError("unknown operator: %s" % op)
            if not ok:
                return None
        return env
    if value is MISSING or type(value) is not type(spec) or value != spec:
        return None
    return env


def match(pattern, step, env=None, bind=True):
    """Match one step against a pattern. Returns the env on success, None on failure."""
    env = {} if env is None else env
    tool = pattern.get("tool")
    if tool is not None:
        if isinstance(tool, list):
            if step.get("tool") not in tool:
                return None
        elif step.get("tool") != tool:
            return None
    for path, spec in pattern.get("where", {}).items():
        env = _spec_ok(spec, get(step, path), env, bind)
        if env is None:
            return None
    return env


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _visible(earlier, later):
    """Was the result of `earlier` known when `later` was issued?"""
    a, b = earlier.get("turn"), later.get("turn")
    return a is None or b is None or a < b


def _in_window(rule, earlier, later):
    limit = rule.get("within")
    if limit is None:
        return True
    a, b = earlier.get("ts"), later.get("ts")
    if not _num(a) or not _num(b):
        return False
    return 0 <= b - a <= limit


def _describe(rule, env):
    text = rule.get("desc", rule["id"])
    if env:
        text += " (" + ", ".join("%s=%s" % (k[1:], env[k]) for k in sorted(env)) + ")"
    return text


def _precedes(rule, trace, bind):
    out = []
    for i, step in enumerate(trace):
        env = match(rule["then"], step, {}, bind)
        if env is None:
            continue
        envs = [env]
        if "if_seen" in rule:
            envs = [e for e in (match(rule["if_seen"], s, env, bind)
                                for s in trace[:i] if _visible(s, step))
                    if e is not None]
            if not envs:
                continue
        ok = any(match(rule["need"], s, e, bind) is not None
                 for e in envs for s in trace[:i]
                 if _visible(s, step) and _in_window(rule, s, step))
        if not ok:
            out.append(Violation(rule["id"], i, _describe(rule, env)))
    return out


def _responds(rule, trace, bind):
    out = []
    for i, step in enumerate(trace):
        env = match(rule["when"], step, {}, bind)
        if env is None:
            continue
        if not any(match(rule["expect"], s, env, bind) is not None
                   for s in trace[i + 1:] if _in_window(rule, step, s)):
            out.append(Violation(rule["id"], i, _describe(rule, env)))
    return out


def _never(rule, trace, bind):
    out = []
    for i, step in enumerate(trace):
        env = match(rule["match"], step, {}, bind)
        if env is None:
            continue
        if "after" in rule and not any(
                match(rule["after"], s, env, bind) is not None for s in trace[:i]):
            continue
        out.append(Violation(rule["id"], i, _describe(rule, env)))
    return out


def _at_most(rule, trace, bind):
    out = []
    counts = {}
    for i, step in enumerate(trace):
        if "reset" in rule:
            renv = match(rule["reset"], step, {}, bind)
            if renv is not None:
                counts[tuple(sorted(renv.items()))] = 0
                continue
        env = match(rule["match"], step, {}, bind)
        if env is None:
            continue
        key = tuple(sorted(env.items()))
        counts[key] = counts.get(key, 0) + 1
        if counts[key] == rule["n"] + 1:
            out.append(Violation(rule["id"], i, _describe(rule, env)))
    return out


def _grounded(rule, trace, bind):
    out = []
    pattern = re.compile(rule["extract"])
    for i, step in enumerate(trace):
        env = match(rule["match"], step, {}, bind)
        if env is None:
            continue
        text = get(step, rule["field"])
        if not isinstance(text, str):
            continue
        quoted = [m.group(0) for m in pattern.finditer(text)]
        if not quoted:
            continue
        sources = [s for s in trace[:i]
                   if _visible(s, step) and match(rule["sources"], s, env, bind) is not None]
        if rule.get("latest"):
            sources = sources[-1:]
        seen = " ".join(json.dumps(s.get("result"), sort_keys=True) for s in sources)
        missing = [q for q in quoted if q not in seen]
        if missing:
            out.append(Violation(rule["id"], i, "%s: %s" % (
                _describe(rule, env), ", ".join(repr(q) for q in missing))))
    return out


_KINDS = {"precedes": _precedes, "responds": _responds, "never": _never, "at_most": _at_most,
          "grounded": _grounded}
_REQUIRED = {"precedes": ("then", "need"), "responds": ("when", "expect"),
             "never": ("match",), "at_most": ("match", "n"),
             "grounded": ("match", "field", "extract", "sources")}


def validate(rules):
    """Raise ValueError on a malformed rule file. Returns the rules unchanged."""
    seen = set()
    for rule in rules:
        rid = rule.get("id")
        if not rid or rid in seen:
            raise ValueError("rule id missing or repeated: %r" % rid)
        seen.add(rid)
        kind = rule.get("type")
        if kind not in _KINDS:
            raise ValueError("%s: unknown rule type %r" % (rid, kind))
        for key in _REQUIRED[kind]:
            if key not in rule:
                raise ValueError("%s: missing %r" % (rid, key))
        if "within" in rule:
            if kind not in ("precedes", "responds"):
                raise ValueError("%s: within only applies to precedes and responds" % rid)
            if not _num(rule["within"]) or rule["within"] < 0:
                raise ValueError("%s: within must be a number of seconds >= 0" % rid)
        if kind == "grounded":
            try:
                re.compile(rule["extract"])
            except (re.error, TypeError) as e:
                raise ValueError("%s: bad extract regex: %s" % (rid, e))
        if kind == "at_most" and (not isinstance(rule["n"], int) or isinstance(rule["n"], bool)
                                  or rule["n"] < 0):
            raise ValueError("%s: n must be an integer >= 0" % rid)
        if kind == "at_most" and "reset" in rule:
            mvars = _vars(rule["match"])
            if _vars(rule["reset"]) != mvars:
                raise ValueError("%s: reset must bind the same variables as match" % rid)
    return rules


def _vars(pattern):
    return {v for v in pattern.get("where", {}).values()
            if isinstance(v, str) and v.startswith("$")}


def check(trace, rules, bind=True):
    """Return all violations, ordered by step index. bind=False ignores variables."""
    out = []
    for rule in rules:
        out.extend(_KINDS[rule["type"]](rule, trace, bind))
    out.sort(key=lambda v: (v.step, v.rule))
    return out


def load_rules(path):
    with open(path) as f:
        return validate(json.load(f))


def load_trace(path):
    """Load a trace file. Returns (steps, problems).

    Accepts a JSON list of steps, JSON lines of steps, or a chat-completions style
    message log (a list of {"role": ...} messages), which goes through adapters.from_chat.
    """
    with open(path) as f:
        text = f.read().strip()
    if text.startswith("["):
        data = json.loads(text)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if data and all(isinstance(m, dict) and "role" in m for m in data):
        import adapters
        return adapters.from_chat(data)
    bad = [i for i, s in enumerate(data) if not isinstance(s, dict) or "tool" not in s]
    if bad:
        raise ValueError("step %d is not a tool-call step" % bad[0])
    return data, []


def main(argv):
    if len(argv) != 3:
        print("usage: python tracecheck.py rules.json trace.json")
        return 2
    try:
        rules = load_rules(argv[1])
        trace, problems = load_trace(argv[2])
    except (OSError, ValueError) as e:
        print("error: %s" % e)
        return 2
    found = check(trace, rules)
    for p in problems:
        print("log problem  %s  %s" % (p["kind"], p["detail"]))
    for v in found:
        print("step %d  %s  %s  [%s]" % (v.step, trace[v.step].get("tool"), v.message, v.rule))
    print("%d steps, %d rules, %d violations, %d log problems" % (
        len(trace), len(rules), len(found), len(problems)))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
