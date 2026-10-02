# Allowed values the model must choose from. Defining them here keeps the
# schema, the prompt and the evaluation in agreement.
SEVERITY_LEVELS = ["low", "medium", "high"]

ROOT_CAUSE_CATEGORIES = [
    "driver error",
    "signal visibility",
    "braking distance misjudged",
    "distraction",
    "rail fatigue",
    "deferred maintenance",
    "ground movement",
    "manufacturing defect",
    "equipment failure",
    "public misuse",
    "barrier timing",
    "poor sightlines",
    "component wear",
    "software fault",
    "maintenance overdue",
    "sensor failure",
    "wet surface",
    "poor lighting",
    "damaged flooring",
    "obstruction",
    "other",
]
