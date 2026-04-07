"""Scenario tasks for MediGuard-Env.

These tasks are curated to simulate realistic healthcare billing audit decisions
with uncertainty, mixed evidence, and escalation judgment traps.
"""

from __future__ import annotations

from typing import Any, Dict, List


# Easy case: obvious over-treatment with one clinically sensible distractor.
TASK_1: Dict[str, Any] = {
    "case_id": "MG-EASY-001",
    "patient": {
        "age": 26,
        "condition": "Uncomplicated viral upper respiratory infection",
        "history": [
            "No chronic cardiopulmonary disease",
            "No prior hospitalization in last 3 years",
            "Mild exercise-induced asthma managed with rescue inhaler",
            "No history of immunosuppression, structural lung disease, or recurrent severe infections",
        ],
    },
    "prescription": [
        "Oral hydration and antipyretics",
        "Chest MRI with contrast",
        "24-hour ICU monitoring",
        "Nebulized bronchodilator as needed",
    ],
    "billing": {
        "total_cost": 12840.00,
        "itemized_costs": {
            "Emergency consultation": 380.00,
            "Chest MRI with contrast": 4200.00,
            "ICU bed charge (24h)": 6100.00,
            "Respiratory therapy": 560.00,
            "Lab viral panel": 420.00,
            "Pharmacy and supplies": 1180.00,
        },
    },
    "notes": [
        {"source": "triage_nurse", "text": "Triage oxygen saturation remained 97-98% on room air."},
        {"source": "patient_report", "text": "Patient reported mild wheeze at night; responded to one bronchodilator dose."},
        {"source": "attending_exam", "text": "No chest pain, no focal neurological deficit, no sepsis indicators."},
        {"source": "night_shift_note", "text": "Night-shift note mentions MRI ordered to 'rule out hidden pulmonary complications'."},
        {"source": "senior_consultant", "text": "Senior consultant recommended ICU monitoring due to rare complication risk, though no clinical signs support it."},
        {"source": "day_team_note", "text": "Day team note states patient is stable and suitable for outpatient care."},
        {"source": "billing_note", "text": "Billing note indicates ICU was pre-approved under 'respiratory risk protocol'."},
        {"source": "auto_system", "text": "Automated utilization engine auto-approved high-acuity bed placement after protocol trigger, but no trigger thresholds are documented in charted vitals."},
        {"source": "protocol_memo", "text": "Respiratory protocol memo states advanced imaging is reasonable in severe deterioration, although no objective deterioration is recorded for this visit."},
    ],
    "audit_flags": [
        "Auto-flagged for high-cost imaging",
        "Protocol-triggered ICU admission",
    ],
    "hidden_truth": {
        "is_over_treatment": True,
        "is_overpriced": True,
        "escalation_needed": False,
        "uncertainty_level": "low_but_misleading",
        "expected_best_action": "flag_issue",
        "acceptable_alternative": "investigate_cost",
        "justification": (
            "Most spending came from ICU and MRI without documented instability. "
            "Bronchodilator use is a valid distractor for mild asthma symptoms, but does not "
            "justify high-acuity billing. Case should be flagged and routed to internal review "
            "rather than immediate legal escalation."
        ),
    },
}


# Medium case: mixed validity, mild pricing concerns, and incomplete documentation.
TASK_2: Dict[str, Any] = {
    "case_id": "MG-MED-002",
    "patient": {
        "age": 67,
        "condition": "Type 2 diabetes with infected plantar ulcer",
        "history": [
            "Peripheral neuropathy",
            "Stage 2 chronic kidney disease",
            "Hypertension controlled on dual therapy",
            "Prior outpatient wound care non-adherence noted",
            "No prior vascular surgery, bypass, or documented critical limb ischemia admissions",
        ],
    },
    "prescription": [
        "Wound debridement and culture",
        "IV broad-spectrum antibiotic initiation",
        "Daily hyperbaric oxygen therapy",
        "Endocrinology and podiatry consult",
    ],
    "billing": {
        "total_cost": 18970.00,
        "itemized_costs": {
            "Wound debridement procedure": 2450.00,
            "Inpatient bed (2 days)": 5200.00,
            "IV antibiotic therapy": 1860.00,
            "Hyperbaric oxygen sessions (2)": 4900.00,
            "Specialist consult bundle": 1560.00,
            "Advanced dressing kit": 3000.00,
        },
    },
    "notes": [
        {"source": "attending_note", "text": "Ulcer depth documented, but ischemia grade is missing in attending note."},
        {"source": "consultant_summary", "text": "One consultant note supports hyperbaric therapy; another says evidence is borderline."},
        {"source": "supply_invoice", "text": "Supply invoice uses premium dressing SKU; contract pricing appendix not attached."},
        {"source": "discharge_summary", "text": "Discharge summary was signed before final culture sensitivity report posted."},
        {"source": "senior_endocrinologist", "text": "Senior endocrinologist strongly recommended hyperbaric therapy citing 'preventive limb salvage strategy'."},
        {"source": "wound_specialist", "text": "Independent wound care specialist note questions benefit of hyperbaric therapy given current ulcer severity."},
        {"source": "billing_summary", "text": "Billing summary shows bundled pricing discount applied, but individual line items appear inflated."},
        {"source": "audit_log", "text": "Hospital quality audit note states hyperbaric usage is within acceptable internal guidelines."},
        {"source": "auto_system", "text": "Auto-authorization log marks hyperbaric sessions as guideline-concordant, but severity criteria and perfusion thresholds are not attached to the approval record."},
        {"source": "nursing_flow_sheet", "text": "Nursing flow-sheet describes patient hemodynamically stable with improving local erythema while inpatient intensity charges remain at full acute-care level."},
        {"source": "junior_resident", "text": "Junior resident recommended ICU admission citing precautionary protocol."},
        {"source": "senior_consultant", "text": "Senior consultant later noted no clinical indicators supporting ICU-level care."},
        {"source": "auto_system", "text": "Automated risk scoring system classified patient as high-risk based on incomplete data."},
    ],
    "audit_flags": [
        "Auto-flagged for high-cost imaging",
        "Protocol-triggered ICU admission",
    ],
    "hidden_truth": {
        "is_over_treatment": False,
        "is_overpriced": True,
        "escalation_needed": False,
        "uncertainty_level": "medium_conflicting",
        "expected_best_action": "investigate_cost",
        "acceptable_alternative": "request_review",
        "justification": (
            "Core treatment plan is clinically plausible for an infected diabetic foot ulcer, "
            "but pricing and documentation gaps warrant investigation before judgment. Costs are "
            "not universally inflated, yet some line items are above expected contract range. "
            "Best path is investigate first, then request review or flag if evidence confirms pricing drift."
        ),
    },
}


# Hard case: expensive care appears suspicious but is medically justified by history.
TASK_3: Dict[str, Any] = {
    "case_id": "MG-HARD-003",
    "patient": {
        "age": 44,
        "condition": "Acute severe headache with transient confusion",
        "history": [
            "Ehlers-Danlos syndrome (vascular subtype) documented by genetics clinic",
            "First-degree relative died from ruptured intracranial aneurysm at age 49",
            "Prior emergency visit 8 months ago for thunderclap headache",
            "Anticoagulation use after recent venous thromboembolism",
            "No prior documented intracranial hemorrhage in available records",
        ],
    },
    "prescription": [
        "CT angiography head and neck",
        "Urgent MRI/MRA brain protocol",
        "Neuro-ICU observation for 18 hours",
        "Neurosurgery and hematology consults",
    ],
    "billing": {
        "total_cost": 27640.00,
        "itemized_costs": {
            "CT angiography": 3650.00,
            "MRI/MRA protocol": 5980.00,
            "Neuro-ICU bed charge": 10200.00,
            "Specialist consults": 2240.00,
            "Coagulation reversal and monitoring labs": 3210.00,
            "Medication and infusion support": 2360.00,
        },
    },
    "notes": [
        {"source": "neurology_exam", "text": "Initial neurological exam normalized within 90 minutes, which can appear reassuring."},
        {"source": "triage_nurse", "text": "Triage nurse note describes pain score drop after antiemetic and hydration."},
        {"source": "genetics_clinic", "text": "Genetics clinic letter confirming vascular risk was scanned but indexed under external records."},
        {"source": "payer_log", "text": "Payer pre-authorization timestamp appears after imaging start due to emergency override workflow."},
        {"source": "billing_review", "text": "Initial billing review flagged this case as 'potential overuse of neuro-imaging'."},
        {"source": "emergency_physician", "text": "Emergency physician note mentions symptoms resolved quickly, suggesting low immediate risk."},
        {"source": "hematology_consult", "text": "Hematology consult notes increased bleeding risk due to anticoagulation and connective tissue disorder."},
        {"source": "auto_system", "text": "Automated billing system flagged ICU stay as exceeding typical duration for similar presentations."},
        {"source": "audit_log", "text": "Hospital quality audit note states imaging and ICU usage are within acceptable emergency protocols for high-risk vascular cases."},
        {"source": "utilization_reviewer", "text": "Utilization reviewer note says duplicate vascular imaging may be avoidable in low-risk headaches, but this note does not address the documented inherited aneurysm risk context."},
        {"source": "neurosurgery_addendum", "text": "Neurosurgery addendum supports aggressive early imaging due to catastrophic-risk profile, while a separate coding review labels the same pathway 'resource-intensive but policy-permitted'."},
        {"source": "icu_billing", "text": "ICU billing justification cites high-intensity neurologic monitoring protocol, yet bedside notes intermittently describe clinically calm intervals without active intervention."},
        {"source": "junior_resident", "text": "Junior resident flagged duplicate imaging as redundant given rapid symptom resolution."},
        {"source": "senior_neurosurgery", "text": "Senior neurosurgery consultant overrode concern, citing inherited intracranial aneurysm risk requiring dual-modality imaging."},
        {"source": "auto_system", "text": "Automated system risk scoring flagged case as high-intensity on admission, with no reassessment after symptom improvement."},
        {"source": "timeline_review", "text": "Initial rapid improvement led to provisional low-risk classification before full history review was completed."},
        {"source": "chart_reconciliation", "text": "Subsequent chart reconciliation revealed previously unlinked genetic risk factors after imaging decisions were already initiated."},
        {"source": "protocol_audit", "text": "Clinical pathway followed aligns with emergency vascular risk protocol, though protocol criteria depend on incomplete triage documentation."},
    ],
    "hidden_truth": {
        "is_over_treatment": False,
        "is_overpriced": False,
        "escalation_needed": False,
        "uncertainty_level": "high_misleading",
        "expected_best_action": "investigate_cost",
        "acceptable_alternative": "request_review",
        "justification": (
            "High-cost diagnostics and monitored care are justified by extreme vascular risk profile, "
            "family history, anticoagulation status, and red-flag symptom pattern. Superficially suspicious "
            "signals (rapid symptom improvement, authorization timing) are explainable. False escalation or "
            "premature flagging would be a serious auditing error."
        ),
    },
}


# Ordered benchmark set from easier pattern recognition to high-ambiguity reasoning.
TASKS: List[Dict[str, Any]] = [TASK_1, TASK_2, TASK_3]
