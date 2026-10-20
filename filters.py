"""
filters.py - heuristic "is this location in the USA?" check.

This is a string heuristic, not a geocoder. Known misses and risks:
  - "Greater Boston" and other region names are not recognized (false negative)
  - a two-letter ending like ", DE" or ", IN" is read as a US state, so a
    non-US location written that way would be wrongly kept (false positive)
  - a bare "Georgia" is NOT treated as US (it is also a country)
"""

STATE_ABBR = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
}

STATE_NAMES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "hawaii", "idaho", "illinois",
    "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland",
    "massachusetts", "michigan", "minnesota", "mississippi", "missouri",
    "montana", "nebraska", "nevada", "new hampshire", "new jersey",
    "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont", "virginia",
    "washington", "west virginia", "wisconsin", "wyoming",
    "district of columbia",
}


def is_us_location(location):
    s = (location or "").strip()
    if not s:
        return False
    low = s.lower()
    if "united states" in low or low in {"usa", "us", "u.s.", "u.s.a."}:
        return True
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) == 1 and parts[0].lower() in STATE_NAMES:
        return True
    for p in parts[1:]:
        if p.upper() in STATE_ABBR or p.lower() in STATE_NAMES:
            return True
    return False
