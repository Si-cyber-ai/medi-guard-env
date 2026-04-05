You are a senior Python engineer integrating task logic into an OpenEnv environment.

Project:
“MediGuard-Env: AI Healthcare Billing Audit & Legal Escalation Environment”

IMPORTANT:
- Do NOT change overall architecture
- Keep code clean and modular
- Do NOT add reward or grading yet
- Focus ONLY on state transitions and task integration

STEP 4 GOAL:
Integrate tasks from env/tasks.py into MediGuardEnv.

--------------------------------------------------

1. IMPORT TASKS

Import:
from env.tasks import TASKS

--------------------------------------------------

2. UPDATE reset()

Modify reset() so that:

- It selects a task from TASKS
- Use simple cycling or random selection
- Set self.current_case to selected task (excluding hidden_truth)

IMPORTANT:
- Do NOT expose hidden_truth in observation
- Store hidden_truth internally:
    self._hidden_truth = task["hidden_truth"]

--------------------------------------------------

3. PARTIAL OBSERVABILITY

At reset:
- Only expose:
    patient
    condition
    minimal prescription info
    basic billing total

Hide:
- detailed notes
- itemized costs
- justification

--------------------------------------------------

4. ACTION EFFECTS (CRITICAL)

Modify step() so actions change what agent sees:

--------------------------------------------------

analyze_case:
- reveals prescription fully
- reveals basic notes (first 1–2 notes)

--------------------------------------------------

investigate_cost:
- reveals itemized_costs
- reveals cost anomalies (if any)

--------------------------------------------------

check_guidelines:
- reveals more notes
- may reveal hints about justification

--------------------------------------------------

request_review:
- reveals additional hidden notes
- does NOT end episode

--------------------------------------------------

flag_issue / approve_case / escalate_case:
- mark episode as done

--------------------------------------------------

5. STATE UPDATES

Track:
- what information is revealed
- what actions have been performed

Example flags:
- analysis_done
- investigation_done
- guidelines_checked (add this)
- review_requested (add this)

--------------------------------------------------

6. OBSERVATION UPDATE

Modify _build_observation() so it reflects:

- progressively revealed data
- not full data from start

Example:
- before analyze → limited info
- after investigate → more details

--------------------------------------------------

7. ADD SAFETY RULE

If agent tries:
- investigate_cost BEFORE analyze_case

Allow it, but:
- do NOT reveal full data
- simulate partial confusion

--------------------------------------------------

8. KEEP CLEAN STRUCTURE

- Use helper methods if needed:
    _apply_action_effects()
    _reveal_data()

- Do NOT mix logic into one big function

--------------------------------------------------

9. DO NOT ADD:

- reward logic
- grading
- inference logic

--------------------------------------------------

After implementing:
Explain how information flow changes across steps.