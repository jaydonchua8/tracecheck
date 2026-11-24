"""Read and write agent tool calls in the chat-completions message format.

The format is the public one used by OpenAI-style chat APIs:

  {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
       "function": {"name": "book_appointment", "arguments": "{\\"slot_id\\": \\"S1\\"}"}}]}
  {"role": "tool", "tool_call_id": "call_1", "content": "{\\"ok\\": true}"}

`arguments` is a JSON string written by the model, so it can be malformed. One
assistant message can carry several calls (parallel tool calls), and results are
paired to calls by id, not by position. The format has no timestamps. If a message
carries a "ts" key (epoch seconds or ISO 8601) I use it; that key is my addition.

from_chat never raises on a bad log. It returns the steps it could recover plus a
list of problems, because a checker that crashes on a truncated log is useless
exactly when something went wrong.
"""
import json
from datetime import datetime, timezone


def _ts(value):
    """Epoch seconds from a number or an ISO 8601 string. None if unreadable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return None


def _content(raw):
    """Tool output as a dict, so rules can use dotted paths into it."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            return {"text": raw}
        return value if isinstance(value, dict) else {"value": value}
    return {"value": raw}


def from_chat(messages):
    """Convert a message log to steps. Returns (steps, problems).

    Each step is {"tool", "args", "turn", "call_id"} plus "result" if a tool
    message answered the call and "ts" if the assistant message had a timestamp.
    A problem is {"kind", "index", "detail"}; index is the message index.
    Kinds: not_a_message, bad_tool_call, bad_arguments, duplicate_call_id,
    orphan_result, duplicate_result, bad_timestamp, no_result.
    """
    steps, problems, by_id = [], [], {}

    def problem(kind, index, detail):
        problems.append({"kind": kind, "index": index, "detail": detail})

    turn = -1
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or "role" not in msg:
            problem("not_a_message", i, "message %d has no role" % i)
            continue
        ts = None
        if "ts" in msg:
            ts = _ts(msg["ts"])
            if ts is None:
                problem("bad_timestamp", i, "message %d: unreadable ts %r" % (i, msg["ts"]))
        calls = msg.get("tool_calls")
        if msg["role"] == "assistant" and calls:
            if not isinstance(calls, list):
                problem("bad_tool_call", i, "message %d: tool_calls is not a list" % i)
                continue
            turn += 1
            for call in calls:
                fn = call.get("function") if isinstance(call, dict) else None
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                    problem("bad_tool_call", i, "message %d: call without a function name" % i)
                    continue
                step = {"tool": fn["name"], "args": {}, "turn": turn, "call_id": call.get("id")}
                if ts is not None:
                    step["ts"] = ts
                raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                    if not isinstance(args, dict):
                        raise ValueError("arguments are not an object")
                    step["args"] = args
                except ValueError:
                    step["args_raw"] = raw
                    problem("bad_arguments", i, "%s (%s): arguments are not a JSON object"
                            % (fn["name"], call.get("id")))
                cid = call.get("id")
                if cid is None or cid in by_id:
                    problem("duplicate_call_id", i, "%s: missing or repeated call id %r"
                            % (fn["name"], cid))
                else:
                    by_id[cid] = step
                steps.append(step)
        elif msg["role"] == "tool":
            cid = msg.get("tool_call_id")
            step = by_id.get(cid)
            if step is None:
                problem("orphan_result", i, "tool message for unknown call id %r" % cid)
            elif "result" in step:
                problem("duplicate_result", i, "second result for call id %r ignored" % cid)
            else:
                step["result"] = _content(msg.get("content"))
    for step in steps:
        if "result" not in step:
            problem("no_result", None, "%s (%s): no tool message answered this call"
                    % (step["tool"], step["call_id"]))
    return steps, problems


def to_chat(steps, shuffle=None):
    """Write steps as a message log. Steps sharing a "turn" become one assistant
    message with parallel tool calls. `shuffle` is an optional random.Random used
    to reorder the tool messages inside a turn. A step without "result" gets no
    tool message."""
    messages = [{"role": "system", "content": "You coordinate referrals for a clinic."}]
    groups = []
    for n, step in enumerate(steps):
        turn = step.get("turn")
        if groups and turn is not None and groups[-1][0] == turn:
            groups[-1][1].append((n, step))
        else:
            groups.append((turn, [(n, step)]))
    for _, group in groups:
        msg = {"role": "assistant", "content": None, "tool_calls": []}
        if "ts" in group[0][1]:
            msg["ts"] = group[0][1]["ts"]
        replies = []
        for n, step in group:
            cid = "call_%04d" % n
            msg["tool_calls"].append({"id": cid, "type": "function", "function": {
                "name": step["tool"], "arguments": json.dumps(step.get("args", {}))}})
            if "result" in step:
                replies.append({"role": "tool", "tool_call_id": cid,
                                "content": json.dumps(step["result"])})
        if shuffle is not None:
            shuffle.shuffle(replies)
        messages.append(msg)
        messages.extend(replies)
    return messages
