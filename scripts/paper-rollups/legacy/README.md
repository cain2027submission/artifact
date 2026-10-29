# frozen paper-rollups scripts

These scripts produced published results and are no longer maintained. They were not moved into this directory because other scripts, `run_all.sh`, data files or notes still reference them by filename or module name, so a move would break an import or a citation. Each is reproducible only at its original commit; do not edit them to track `grading_common.py`.

| script | what it produced |
|---|---|
| `part1_structural_loss.py` | part 1 structural loss accounting, a classification of what rollups structurally cannot contain |
| `part2_streamlined.py` | part 2 question retention with progress reporting, the first RQ3 question-retention run |
| `resume_rq3.py` | resumed RQ3 question generation from where `part2_streamlined.py` stopped, extending the sample to 55-70 sessions |
| `resume_rq3_v2.py` | the same resume with more robust JSON parsing, grades through `grading_common` |
| `part3_dead_end_verification.py` | part 3 dead-end hallucination check, rollup claims verified against transcripts |
| `part3_simple.py` | part 3 dead-end precision, a first variant |
| `part3_streamlined.py` | part 3 dead-end precision, a streamlined variant |

`resume_rq3_v2.py` and `part2_question_retention.py` are the only two that still grade through `grading_common` and so raise `GradeError` on a grader failure. `part2_streamlined.py` and `resume_rq3.py` carry their own grader copies and predate the shared module.
