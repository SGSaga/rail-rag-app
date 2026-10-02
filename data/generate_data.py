"""
Generate synthetic rail safety incident reports.

This is entirely fake data for a learning/portfolio project. It does not
use, reproduce or derive from any real RSSB records. Reports are assembled
from templated components with randomised details.
"""

import json
import random
from datetime import datetime, timedelta

random.seed(42)  # reproducible output

LOCATIONS = [
    "Clapham Junction", "Reading West", "Doncaster", "Crewe", "Bristol Temple Meads",
    "Leeds", "Manchester Piccadilly", "Edinburgh Waverley", "Birmingham New Street",
    "York", "Peterborough", "Swindon", "Didcot", "Rugby", "Carlisle",
]

INCIDENT_TYPES = {
    "signal_passed_at_danger": {
        "phrases": [
            "train passed signal {sig} at danger",
            "SPAD recorded at signal {sig}",
            "driver reported overrunning signal {sig}",
        ],
        "root_causes": ["driver error", "signal visibility", "braking distance misjudged", "distraction"],
        "severity_bias": ["high", "high", "medium"],
    },
    "track_defect": {
        "phrases": [
            "rail defect identified on the {track} line",
            "cracked rail reported near {loc}",
            "track geometry fault detected during inspection",
        ],
        "root_causes": ["rail fatigue", "deferred maintenance", "ground movement", "manufacturing defect"],
        "severity_bias": ["medium", "high", "medium"],
    },
    "level_crossing": {
        "phrases": [
            "barrier failure at {loc} level crossing",
            "pedestrian misuse reported at {loc} crossing",
            "vehicle stranded on level crossing near {loc}",
        ],
        "root_causes": ["equipment failure", "public misuse", "barrier timing", "poor sightlines"],
        "severity_bias": ["high", "medium", "high"],
    },
    "rolling_stock": {
        "phrases": [
            "door fault on unit {sig} in service",
            "brake pressure anomaly reported on service {sig}",
            "traction power loss on {track} line service",
        ],
        "root_causes": ["component wear", "software fault", "maintenance overdue", "sensor failure"],
        "severity_bias": ["medium", "medium", "low"],
    },
    "slips_trips": {
        "phrases": [
            "passenger slip on platform at {loc}",
            "staff member reported trip hazard on concourse at {loc}",
            "fall on stairs reported at {loc} station",
        ],
        "root_causes": ["wet surface", "poor lighting", "damaged flooring", "obstruction"],
        "severity_bias": ["low", "low", "medium"],
    },
}

WEATHER = ["clear", "heavy rain", "fog", "ice", "high winds", "snow"]
TIMES = ["early morning", "morning peak", "midday", "evening peak", "late evening", "overnight"]

def make_report(i, start_date):
    itype = random.choice(list(INCIDENT_TYPES.keys()))
    spec = INCIDENT_TYPES[itype]
    loc = random.choice(LOCATIONS)
    sig = f"{random.choice('ABCDEFGH')}{random.randint(100, 999)}"
    track = random.choice(["up", "down", "relief", "main"])
    phrase = random.choice(spec["phrases"]).format(sig=sig, loc=loc, track=track)
    root = random.choice(spec["root_causes"])
    severity = random.choice(spec["severity_bias"])
    weather = random.choice(WEATHER)
    time_of_day = random.choice(TIMES)
    date = start_date + timedelta(days=random.randint(0, 720))

    narrative = (
        f"On {date.strftime('%d %B %Y')} during the {time_of_day}, {phrase} at or near {loc}. "
        f"Weather at the time was {weather}. "
        f"Initial assessment points to {root} as a contributing factor. "
        f"No {'injuries' if severity != 'high' else 'serious injuries'} were reported "
        f"{'though the incident had potential for harm' if severity == 'high' else 'and services resumed shortly after'}. "
        f"Further investigation was {'recommended' if severity != 'low' else 'not considered necessary'}."
    )

    return {
        "id": f"INC-{2024}-{i:04d}",
        "date": date.strftime("%Y-%m-%d"),
        "location": loc,
        "incident_type": itype,
        "narrative": narrative,
        # ground-truth labels, useful later for evaluating the model
        "true_severity": severity,
        "true_root_cause": root,
    }

def main(n=80):
    start = datetime(2024, 1, 1)
    reports = [make_report(i, start) for i in range(1, n + 1)]
    with open("incidents.json", "w") as f:
        json.dump(reports, f, indent=2)
    print(f"Wrote {len(reports)} synthetic incident reports to incidents.json")
    print("\nExample:\n")
    print(json.dumps(reports[0], indent=2))

if __name__ == "__main__":
    main()
