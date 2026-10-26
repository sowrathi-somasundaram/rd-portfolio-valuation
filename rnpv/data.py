"""Five hypothetical candidates. Costs and durations are illustrative, sized to
published industry ranges; none corresponds to a real asset."""
from .model import Drug

DEFAULT_DRUGS = [
    Drug("ONC-117", "Oncology (ADC, solid tumours)", "Phase I",
         durations=(2, 2, 3, 1), costs=(45, 130, 450, 10), peak_sales=2_000, years_to_peak=6,
         patent_expiry=2043, margin=0.40, loe_erosion=0.50, peak_cv=0.50, delay_prob=0.30),
    Drug("IMM-204", "Immunology (oral IL-23, psoriasis)", "Phase II",
         durations=(1, 2, 3, 1), costs=(30, 90, 350, 8), peak_sales=1_200, years_to_peak=5,
         patent_expiry=2041, margin=0.40, loe_erosion=0.70, peak_cv=0.40, delay_prob=0.25),
    Drug("NEU-310", "Neuroscience (Alzheimer's)", "Phase II",
         durations=(2, 2, 4, 1), costs=(40, 160, 750, 12), peak_sales=3_500, years_to_peak=6,
         patent_expiry=2043, margin=0.40, loe_erosion=0.50, peak_cv=0.60, delay_prob=0.35),
    Drug("RES-402", "Respiratory (inhaled COPD)", "Phase III",
         durations=(1, 2, 3, 1), costs=(25, 70, 300, 8), peak_sales=750, years_to_peak=4,
         patent_expiry=2038, margin=0.35, loe_erosion=0.60, peak_cv=0.30, delay_prob=0.20),
    Drug("VAC-115", "Infectious disease (vaccine)", "Phase I",
         durations=(2, 2, 3, 1), costs=(30, 80, 400, 10), peak_sales=1_500, years_to_peak=5,
         patent_expiry=2045, margin=0.35, loe_erosion=0.30, peak_cv=0.40, delay_prob=0.25),
]

DEFAULT_BUDGET = 800.0
DEFAULT_RATE = 0.10
