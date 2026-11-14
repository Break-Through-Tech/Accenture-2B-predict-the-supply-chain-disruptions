"""Static domain knowledge used by the dashboard.

Everything here is editable plain data: ports, routes, industry sensitivity
profiles and the playbook of workarounds. None of it is learned from your CSV -
the Random Forest only supplies the risk probabilities.
"""

# --------------------------------------------------------------------------
# Model drivers (base feature -> label, unit, which direction is WORSE)
#   +1 = higher value is worse, -1 = lower value is worse
#   NOTE: direction for container_availability_index is an assumption
#   (low availability = trouble). Flip it here if your index works the other way.
# --------------------------------------------------------------------------
DRIVERS = {
    "shipping_delay_days":          ("Shipping delays", "days", +1),
    "port_congestion_index":        ("Port congestion", "index", +1),
    "freight_cost_usd":             ("Freight cost", "USD", +1),
    "container_availability_index": ("Container availability", "index", -1),
    "weather_disruption_score":     ("Weather disruption", "score", +1),
    "geopolitical_risk_score":      ("Geopolitical risk", "score", +1),
    "fuel_cost_index":              ("Fuel cost", "index", +1),
    "commodity_price_index":        ("Commodity prices", "index", +1),
    "oil_price":                    ("Oil price", "USD/bbl", +1),
}

# Drivers the user can override from the sidebar (the rest stay at the seasonal baseline)
EDITABLE_DRIVERS = [
    "shipping_delay_days", "port_congestion_index", "freight_cost_usd",
    "container_availability_index", "weather_disruption_score",
    "geopolitical_risk_score", "fuel_cost_index", "commodity_price_index",
]

# --------------------------------------------------------------------------
# Risk tier colours: validated status palette (icon + label always shown too)
# --------------------------------------------------------------------------
TIERS = [
    # name, icon, colour
    ("Low",      "●", "#0ca30c"),
    ("Moderate", "▲", "#fab219"),
    ("Elevated", "◆", "#ec835a"),
    ("High",     "■", "#d03b3b"),
]
TIER_COLOR = {t[0]: t[2] for t in TIERS}
TIER_ICON = {t[0]: t[1] for t in TIERS}

# --------------------------------------------------------------------------
# US ports  (name, lat, lon, coast, hurricane-exposed)
# --------------------------------------------------------------------------
PORTS = [
    ("Los Angeles / Long Beach", 33.74, -118.26, "West", False),
    ("Oakland",                  37.80, -122.30, "West", False),
    ("Seattle / Tacoma",         47.27, -122.40, "West", False),
    ("Houston",                  29.73,  -95.27, "Gulf", True),
    ("New Orleans",              29.93,  -90.06, "Gulf", True),
    ("Mobile",                   30.69,  -88.04, "Gulf", True),
    ("Tampa",                    27.94,  -82.45, "Gulf", True),
    ("Miami",                    25.77,  -80.17, "East", True),
    ("Jacksonville",             30.39,  -81.55, "East", True),
    ("Savannah",                 32.08,  -81.09, "East", True),
    ("Charleston",               32.78,  -79.93, "East", True),
    ("Norfolk",                  36.85,  -76.30, "East", False),
    ("Baltimore",                39.26,  -76.58, "East", False),
    ("New York / New Jersey",    40.67,  -74.05, "East", False),
    ("Boston",                   42.35,  -71.04, "East", False),
]

# --------------------------------------------------------------------------
# State centroids (lat, lon) - used to find the nearest gateway ports
# --------------------------------------------------------------------------
STATES = {
    "AL": ("Alabama", 32.8, -86.8), "AK": ("Alaska", 61.4, -152.3), "AZ": ("Arizona", 34.2, -111.7),
    "AR": ("Arkansas", 34.9, -92.4), "CA": ("California", 37.2, -119.5), "CO": ("Colorado", 39.0, -105.5),
    "CT": ("Connecticut", 41.6, -72.7), "DE": ("Delaware", 39.0, -75.5), "DC": ("District of Columbia", 38.9, -77.0),
    "FL": ("Florida", 28.6, -82.4), "GA": ("Georgia", 32.7, -83.4), "HI": ("Hawaii", 20.8, -156.3),
    "ID": ("Idaho", 44.4, -114.6), "IL": ("Illinois", 40.0, -89.2), "IN": ("Indiana", 39.9, -86.3),
    "IA": ("Iowa", 42.1, -93.5), "KS": ("Kansas", 38.5, -98.4), "KY": ("Kentucky", 37.5, -85.3),
    "LA": ("Louisiana", 31.1, -92.0), "ME": ("Maine", 45.3, -69.2), "MD": ("Maryland", 39.0, -76.8),
    "MA": ("Massachusetts", 42.3, -71.8), "MI": ("Michigan", 44.3, -85.4), "MN": ("Minnesota", 46.3, -94.3),
    "MS": ("Mississippi", 32.7, -89.7), "MO": ("Missouri", 38.4, -92.5), "MT": ("Montana", 47.0, -109.6),
    "NE": ("Nebraska", 41.5, -99.8), "NV": ("Nevada", 39.3, -116.6), "NH": ("New Hampshire", 43.7, -71.6),
    "NJ": ("New Jersey", 40.1, -74.7), "NM": ("New Mexico", 34.4, -106.1), "NY": ("New York", 42.9, -75.5),
    "NC": ("North Carolina", 35.6, -79.4), "ND": ("North Dakota", 47.5, -100.5), "OH": ("Ohio", 40.3, -82.8),
    "OK": ("Oklahoma", 35.6, -97.5), "OR": ("Oregon", 43.9, -120.6), "PA": ("Pennsylvania", 40.9, -77.8),
    "RI": ("Rhode Island", 41.7, -71.5), "SC": ("South Carolina", 33.9, -80.9), "SD": ("South Dakota", 44.4, -100.2),
    "TN": ("Tennessee", 35.9, -86.4), "TX": ("Texas", 31.5, -99.3), "UT": ("Utah", 39.3, -111.7),
    "VT": ("Vermont", 44.1, -72.7), "VA": ("Virginia", 37.5, -78.8), "WA": ("Washington", 47.4, -120.5),
    "WV": ("West Virginia", 38.6, -80.6), "WI": ("Wisconsin", 44.6, -89.9), "WY": ("Wyoming", 43.0, -107.5),
}

# --------------------------------------------------------------------------
# Shipping routes by origin region (matched on a lower-case keyword of the
# region name found in your dataset's origin_region_* columns)
#   coasts: which US coasts the route lands on ("Any" = unrestricted)
#   chokepoints: tags used for route-specific alerts
# --------------------------------------------------------------------------
ROUTES = {
    "east asia": [
        {"name": "Trans-Pacific → US West Coast", "coasts": ["West"], "chokepoints": ["pacific"]},
        {"name": "Trans-Pacific → Panama Canal → US East / Gulf", "coasts": ["East", "Gulf"], "chokepoints": ["panama"]},
        {"name": "Trans-Pacific → West Coast + inland rail", "coasts": ["West"], "chokepoints": ["pacific", "rail"]},
    ],
    "south asia": [
        {"name": "Indian Ocean → Suez / Red Sea → US East Coast", "coasts": ["East"], "chokepoints": ["suez"]},
        {"name": "Indian Ocean → Cape of Good Hope → US East Coast", "coasts": ["East"], "chokepoints": ["cape"]},
        {"name": "Via Southeast Asia → Trans-Pacific → US West Coast", "coasts": ["West"], "chokepoints": ["pacific"]},
    ],
    "middle east": [
        {"name": "Red Sea / Suez → Mediterranean → US East Coast", "coasts": ["East"], "chokepoints": ["suez"]},
        {"name": "Gulf → Cape of Good Hope → US East / Gulf", "coasts": ["East", "Gulf"], "chokepoints": ["cape"]},
        {"name": "Mediterranean → Atlantic → US East Coast", "coasts": ["East"], "chokepoints": []},
    ],
    "europe": [
        {"name": "North Atlantic → US East Coast", "coasts": ["East"], "chokepoints": []},
        {"name": "Atlantic → US Gulf Coast", "coasts": ["Gulf", "East"], "chokepoints": []},
        {"name": "Atlantic → Panama Canal → US West Coast", "coasts": ["West"], "chokepoints": ["panama"]},
    ],
    "latin america": [
        {"name": "Caribbean / Gulf direct → US Gulf & East", "coasts": ["Gulf", "East"], "chokepoints": []},
        {"name": "Pacific coast → US West Coast", "coasts": ["West"], "chokepoints": []},
        {"name": "Overland truck / rail via US-Mexico border", "coasts": ["Any"], "chokepoints": ["border"]},
    ],
    "africa": [
        {"name": "Trans-Atlantic → US East / Gulf", "coasts": ["East", "Gulf"], "chokepoints": []},
        {"name": "Cape of Good Hope → Panama Canal → US West Coast", "coasts": ["West"], "chokepoints": ["cape", "panama"]},
    ],
    "north america": [
        {"name": "Cross-border truck / rail", "coasts": ["Any"], "chokepoints": ["border"]},
        {"name": "Coastal short-sea shipping", "coasts": ["Any"], "chokepoints": []},
    ],
}
DEFAULT_ROUTES = [{"name": "Direct ocean route", "coasts": ["Any"], "chokepoints": []}]

CHOKEPOINT_NOTES = {
    "suez": "Suez / Red Sea transit: exposed to regional security events and carrier re-routing.",
    "cape": "Cape of Good Hope diversion adds roughly 1–2 weeks of transit and fuel burn.",
    "panama": "Panama Canal: draft and transit-slot restrictions are common in the dry season (roughly Dec–Apr).",
    "border": "Border crossing: customs inspection, trucker capacity and policy changes drive delays.",
    "rail": "Inland rail leg: ramp congestion, chassis shortages and weather can add dwell time.",
    "pacific": "Trans-Pacific lane: sensitive to peak-season capacity and West Coast terminal congestion.",
}

# --------------------------------------------------------------------------
# Industry sensitivity profiles.  weight > 1 = that driver matters MORE for the
# industry, so alerts fire earlier (lower percentile threshold) and the driver is
# listed first.  These weights are judgement calls, not fitted from your data.
# --------------------------------------------------------------------------
INDUSTRIES = {
    "Electronics & semiconductors": {
        "keywords": ["electronic", "semiconductor", "tech", "computer"],
        "weights": {"geopolitical_risk_score": 1.4, "container_availability_index": 1.2, "shipping_delay_days": 1.2},
        "extra": [
            "Hold buffer stock of long-lead and single-source components.",
            "Qualify a second source in a different region for critical parts.",
            "Air-freight only the highest-value / line-down items.",
        ],
    },
    "Automotive": {
        "keywords": ["auto", "vehicle", "car"],
        "weights": {"shipping_delay_days": 1.4, "port_congestion_index": 1.2, "geopolitical_risk_score": 1.2},
        "extra": [
            "Move to a JIT-with-buffer model: 5–10 days of safety stock on line-critical parts.",
            "Pre-book rail ramp / drayage slots to protect line-side delivery windows.",
        ],
    },
    "Food & perishables": {
        "keywords": ["food", "perish", "fresh", "beverage"],
        "weights": {"shipping_delay_days": 1.5, "weather_disruption_score": 1.3, "port_congestion_index": 1.3, "fuel_cost_index": 1.1},
        "extra": [
            "Secure reefer containers and plug-in capacity early; avoid long port dwell.",
            "Shift highest-spoilage SKUs to faster modes (air or expedited ocean).",
            "Add a temperature-monitoring and claims-ready cold-chain protocol.",
        ],
    },
    "Pharma & healthcare": {
        "keywords": ["pharma", "health", "medical", "drug"],
        "weights": {"shipping_delay_days": 1.4, "weather_disruption_score": 1.3, "geopolitical_risk_score": 1.1},
        "extra": [
            "Use temperature-controlled lanes with a pre-approved backup carrier.",
            "Split shipments across two lanes so one disruption cannot stop supply.",
        ],
    },
    "Apparel & retail": {
        "keywords": ["apparel", "retail", "textile", "cloth", "fashion"],
        "weights": {"port_congestion_index": 1.3, "freight_cost_usd": 1.3, "container_availability_index": 1.2},
        "extra": [
            "Pull seasonal orders forward and ship before peak-season surcharges.",
            "Negotiate contract rates (not spot) and lock in container allocations.",
        ],
    },
    "Chemicals & plastics": {
        "keywords": ["chem", "plastic", "polymer"],
        "weights": {"weather_disruption_score": 1.3, "oil_price": 1.3, "fuel_cost_index": 1.2, "geopolitical_risk_score": 1.1},
        "extra": [
            "Confirm hazmat-capable routing and carriers; hazmat limits cut re-route options.",
            "Index contracts to feedstock/oil prices to cap margin shocks.",
        ],
    },
    "Energy & fuels": {
        "keywords": ["energy", "fuel", "oil", "gas", "petro"],
        "weights": {"oil_price": 1.5, "geopolitical_risk_score": 1.4, "fuel_cost_index": 1.3, "weather_disruption_score": 1.2},
        "extra": [
            "Hedge price exposure and diversify supply origins.",
            "Pre-arrange alternative terminals / storage for weather-driven closures.",
        ],
    },
    "Machinery & industrial equipment": {
        "keywords": ["machin", "industrial", "equipment", "manufactur"],
        "weights": {"freight_cost_usd": 1.2, "shipping_delay_days": 1.1, "commodity_price_index": 1.2},
        "extra": [
            "Use break-bulk / project-cargo carriers early; capacity is thin and lead times long.",
            "Lock in steel / metals pricing with fixed-price supply agreements.",
        ],
    },
    "Agriculture & commodities": {
        "keywords": ["agri", "commodit", "grain", "crop", "farm"],
        "weights": {"weather_disruption_score": 1.4, "commodity_price_index": 1.4, "fuel_cost_index": 1.1},
        "extra": [
            "Diversify origin ports and use barge / rail alternatives for bulk.",
            "Use forward contracts and storage to smooth price and timing swings.",
        ],
    },
    "Consumer goods (general)": {
        "keywords": [],
        "weights": {},
        "extra": ["Review safety-stock levels by SKU criticality and lead-time variability."],
    },
}
DEFAULT_INDUSTRY = "Consumer goods (general)"

# --------------------------------------------------------------------------
# Workaround playbook, keyed by driver.  When a driver is flagged, these show up.
# --------------------------------------------------------------------------
WORKAROUNDS = {
    "port_congestion_index": [
        "Reroute to a lower-congestion gateway port (see the Port alternatives tab and the map).",
        "Book earlier vessels and pre-arrange chassis, drayage and off-peak gate appointments.",
        "Use inland intermodal ramps or direct rail to bypass terminal yard dwell.",
        "Pre-clear customs and use a Foreign Trade Zone to cut dwell time.",
    ],
    "shipping_delay_days": [
        "Increase safety stock to cover the extra transit variability (add the delay excess in days).",
        "Split the shipment across two carriers or vessels so one delay does not stop everything.",
        "Ask carriers for guaranteed-space or expedited service on time-critical orders.",
        "Move the order date earlier so arrival lands before demand peaks.",
    ],
    "freight_cost_usd": [
        "Shift volume from spot to contract rates, or negotiate a fixed-rate window.",
        "Consolidate shipments (fewer, fuller containers) and use NVOCC / LCL consolidation.",
        "Evaluate a slower, cheaper service level for non-urgent SKUs.",
    ],
    "container_availability_index": [
        "Reserve containers and equipment early; confirm empty-pickup dates at origin.",
        "Use carrier-owned or shipper-owned containers where possible.",
        "Consider alternative equipment (e.g. 40' HC vs 20') or air freight for small urgent lots.",
    ],
    "weather_disruption_score": [
        "Re-time sailings and inland moves away from the storm / freeze window.",
        "Route through a port less exposed to the current weather pattern.",
        "Pre-position inventory at a regional DC before the weather window.",
        "Add weather-delay clauses to carrier agreements and insure cargo.",
    ],
    "geopolitical_risk_score": [
        "Pick an alternate route that avoids the stressed chokepoint (e.g. Cape vs Suez, or West vs East Coast landing).",
        "Qualify alternative origin countries or suppliers for critical SKUs.",
        "Check export-control, sanctions and tariff changes before booking; adjust Incoterms.",
        "Increase cargo insurance cover (war-risk) for affected routes.",
    ],
    "fuel_cost_index": [
        "Negotiate fuel-surcharge caps or index-linked clauses with carriers.",
        "Choose slow-steaming services to cut fuel burn if timing allows.",
    ],
    "commodity_price_index": [
        "Lock in forward prices or fixed-price supply agreements for key inputs.",
        "Buy ahead if prices are trending up and storage cost is acceptable.",
    ],
    "oil_price": [
        "Hedge fuel exposure or switch to index-linked surcharges.",
        "Favour modes and routes with lower fuel intensity (rail, shorter ocean legs).",
    ],
}

GENERIC_HIGH_RISK = [
    "Activate the contingency plan: confirm a backup carrier, backup port and backup supplier now.",
    "Raise safety stock for line-critical SKUs and brief customers on possible delay.",
    "Set daily tracking alerts on vessel ETA, terminal dwell and carrier advisories.",
]

# Lunar New Year dates (factory shutdown + blank sailings)
LUNAR_NEW_YEAR = {
    2024: (2, 10), 2025: (1, 29), 2026: (2, 17), 2027: (2, 6),
    2028: (1, 26), 2029: (2, 13), 2030: (2, 3), 2031: (1, 23),
}
