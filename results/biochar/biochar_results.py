import numpy as np

IMAGE_REGIONS = [
    "CAN", "USA", "MEX", "RCAM", "BRA", "RSAM", "NAF", "WAF", "EAF", "SAF",
    "WEU", "CEU", "TUR", "UKR", "STAN", "RUS", "ME", "INDIA", "KOR", "CHN",
    "SEAS", "INDO", "JAP", "OCE", "RSAS", "RSAF",
]

R10 = {
    "North America": ["CAN", "USA"],
    "Latin America": ["MEX", "RCAM", "BRA", "RSAM"],
    "Europe": ["WEU", "CEU", "TUR"],
    "Eurasia": ["UKR", "STAN", "RUS"],
    "Middle East & North Africa": ["NAF", "ME"],
    "Sub-Saharan Africa": ["WAF", "EAF", "SAF", "RSAF"],
    "South Asia": ["INDIA", "RSAS"],
    "East Asia": ["CHN", "KOR", "JAP"],
    "Southeast Asia": ["SEAS", "INDO"],
    "Oceania": ["OCE"],
}


def aggregate_r26_to_r10_sum(values_r26):
    out = []

    for _, regs in R10.items():
        idx = [IMAGE_REGIONS.index(r) for r in regs]
        out.append(np.nansum(values_r26[idx]))

    return np.asarray(out, dtype=np.float64)