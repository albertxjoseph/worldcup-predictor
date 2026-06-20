"""Static reference data for the 2026 World Cup: hosts, venues, country
coordinates, and helpers shared by the model, simulator, and UI."""

import math
import pandas as pd
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# 2026 host nations, spelled as they appear in the results dataset.
HOSTS = {"United States", "Canada", "Mexico"}

# Official 2026 final group draw (5 Dec 2025), spelled as in the results dataset.
# Keyed by official group letter so the knockout bracket skeleton lines up.
OFFICIAL_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# 2026 venues keyed by the dataset's `city` spelling.
# (latitude, longitude, capacity, country)
VENUES = {
    "Atlanta":         (33.7554, -84.4008, 71000, "United States"),
    "Foxborough":      (42.0909, -71.2643, 65000, "United States"),
    "Arlington":       (32.7473, -97.0945, 80000, "United States"),
    "Houston":         (29.6847, -95.4107, 72000, "United States"),
    "Kansas City":     (39.0490, -94.4839, 76000, "United States"),
    "Inglewood":       (33.9535, -118.3392, 70000, "United States"),
    "Miami Gardens":   (25.9580, -80.2389, 65000, "United States"),
    "East Rutherford": (40.8136, -74.0744, 82500, "United States"),
    "Philadelphia":    (39.9008, -75.1675, 69000, "United States"),
    "Santa Clara":     (37.4030, -121.9700, 68500, "United States"),
    "Seattle":         (47.5952, -122.3316, 69000, "United States"),
    "Mexico City":     (19.3029, -99.1505, 87000, "Mexico"),
    "Guadalupe":       (25.6690, -100.2440, 53500, "Mexico"),   # Monterrey metro
    "Zapopan":         (20.6818, -103.4630, 49800, "Mexico"),   # Guadalajara metro
    "Toronto":         (43.6332, -79.4185, 45000, "Canada"),
    "Vancouver":       (49.2768, -123.1120, 54500, "Canada"),
}

# Team/country names that don't match the country-centroid file, with coords.
COORD_ALIASES = {
    "Curaçao":      (12.1696, -68.9900),
    "DR Congo":     (-4.0383, 21.7587),
    "England":      (52.3555, -1.1743),
    "Ivory Coast":  (7.5400, -5.5471),
    "Scotland":     (56.4907, -4.2026),
    "South Korea":  (35.9078, 127.7669),
    "Iran":         (32.4279, 53.6880),
    "Russia":       (61.5240, 105.3188),
    "Cape Verde":   (16.5388, -23.0418),
    "Cabo Verde":   (16.5388, -23.0418),
}


def load_country_coords():
    """name -> (lat, lon) for as many nations as we can resolve."""
    df = pd.read_csv(os.path.join(HERE, "data", "country_coords.csv"))
    coords = {row["name"]: (row["latitude"], row["longitude"])
              for _, row in df.iterrows()}
    coords.update(COORD_ALIASES)
    return coords


def haversine(a, b):
    """Great-circle distance in km between (lat, lon) points a and b."""
    if a is None or b is None:
        return None
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def derive_groups(wc_fixtures):
    """Recover the 12 groups from the group-stage fixture list: any two teams
    that face each other in the group phase share a group. Returns
    {group_label: [team, ...]} with labels A..L."""
    # Build adjacency from the 72 group matches (knockouts aren't scheduled yet).
    from collections import defaultdict
    adj = defaultdict(set)
    teams = set()
    for _, r in wc_fixtures.iterrows():
        h, a = r["home_team"], r["away_team"]
        adj[h].add(a)
        adj[a].add(h)
        teams.add(h)
        teams.add(a)

    seen = set()
    groups = []
    for t in sorted(teams):
        if t in seen:
            continue
        # connected component = one group of 4
        stack, comp = [t], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            for y in adj[x]:
                if y not in comp:
                    stack.append(y)
        seen |= comp
        groups.append(sorted(comp))

    groups.sort(key=lambda g: g[0])
    labels = [chr(ord("A") + i) for i in range(len(groups))]
    return {labels[i]: groups[i] for i in range(len(groups))}
