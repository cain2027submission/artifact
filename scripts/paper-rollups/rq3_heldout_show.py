#!/usr/bin/env python3
"""Print the full question, ground truth, both arms' raw replies and both grades for
named held-out records. This is the smoke-test evidence the brief asks for: a judge
that silently returned ABSENT for everything, or a grader that never read the
answer, is invisible in a rate and obvious here.

Usage:
    rq3_heldout_show.py <prefix>:<class> [<prefix>:<class> ...]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_raw_transcript_arm import DATA, load_records  # noqa: E402

FILES = {
    "action_record": DATA / "rq3_heldout_action_record_arm.jsonl",
    "output_stripped": DATA / "rq3_heldout_stripped_arm.jsonl",
}


def main() -> None:
    wanted = [tuple(a.split(":", 1)) for a in sys.argv[1:]]
    rows = {arm: {(r["session_prefix"], r["class"]): r for r in load_records(p)}
            for arm, p in FILES.items()}
    for key in wanted:
        a = rows["action_record"].get(key)
        s = rows["output_stripped"].get(key)
        if not a or not s:
            print(f"!! {key} missing in "
                  f"{[arm for arm in FILES if key not in rows[arm]]}")
            continue
        print("=" * 78)
        print(f"SESSION {key[0]}   CLASS {key[1]}   record_id {a['record_id']}")
        print(f"\nQUESTION\n  {a['question']}")
        print(f"\nGROUND TRUTH\n  {a['ground_truth']}")
        for arm, r in (("ACTION RECORD", a), ("OUTPUT-STRIPPED", s)):
            print(f"\n--- {arm}  artifact {r['artifact_chars']:,} chars  "
                  f"prompt {r['usage'].get('prompt_tokens', 0):,} tok")
            print(f"RAW JUDGE REPLY (verbatim, {len(r['raw_reply'] or '')} chars):")
            print("  " + (r["raw_reply"] or "<empty>").replace("\n", "\n  "))
            print(f"EXTRACTED rating={r['rating']}   GRADE={r['grade']}")
        print(f"\nPAIR: action {a['grade']}  vs  stripped {s['grade']}  "
              f"-> {'DISCORDANT' if (a['grade'] in ('correct','partially_correct')) != (s['grade'] in ('correct','partially_correct')) else 'concordant'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
