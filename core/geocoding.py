import requests
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from geopy.geocoders import Nominatim

# ── Single shared geolocator instance ────────────────────────────────────────
geolocator = Nominatim(user_agent="home_tour_planner")

# ════════════════════════════════════════════════════════════════════════════
# GEOCODING METHODS
# ════════════════════════════════════════════════════════════════════════════

def geocode_census(address):
    """
    Method 1: US Census geocoder.
    Free, no API key, results are saveable.
    Works well for established addresses; may miss streets built after 2020.
    """
    url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
    params = {"address": address, "benchmark": "2020", "format": "json"}
    try:
        r = requests.get(url, params=params, timeout=10).json()
        matches = r["result"]["addressMatches"]
        if matches:
            coords = matches[0]["coordinates"]
            return coords["y"], coords["x"]
    except Exception as e:
        print(f"    Census error: {e}")
    return None, None

def geocode_nominatim(address):
    """
    Method 2: Nominatim / OpenStreetMap.
    Free, no API key, community-maintained — catches newer streets Census misses.
    Requires 1 second between requests per usage policy.
    """
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"    Nominatim error: {e}")
    return None, None

def geocode_depot(address):
    """
    Geocode a depot (start/end) address using Census → Nominatim fallback.
    Raises ValueError if both methods fail.
    """
    lat, lon = geocode_census(address)
    time.sleep(0.3)
    if lat is None:
        print(f"⚠️  Census failed for depot, trying Nominatim: {address}")
        lat, lon = geocode_nominatim(address)
        time.sleep(1)
    if lat is None:
        raise ValueError(f"Could not geocode depot address: {address}")
    return lat, lon

# ════════════════════════════════════════════════════════════════════════════
# TRILATERATION (Method 3)
# ════════════════════════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2):
    """Straight-line distance in miles between two lat/lon points."""
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def trilaterate(school_coords, distances):
    """
    Estimate lat/lon from 3 known reference points and distances to each.

    Parameters
    ----------
    school_coords : list of (lat, lon) tuples
    distances     : list of floats — straight-line distances in miles (from Redfin)

    Returns
    -------
    (lat, lon) tuple, or (None, None) if optimization fails
    """
    init_lat = np.mean([c[0] for c in school_coords])
    init_lon = np.mean([c[1] for c in school_coords])

    def objective(point):
        lat, lon = point
        return sum(
            (haversine(lat, lon, slat, slon) - d) ** 2
            for (slat, slon), d in zip(school_coords, distances)
        )

    result = minimize(
        objective,
        x0=[init_lat, init_lon],
        method='Nelder-Mead',
        options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 10000}
    )

    if not result.success and result.fun > 1.0:
        return None, None

    est_lat, est_lon = result.x
    residuals = [
        abs(haversine(est_lat, est_lon, slat, slon) - d)
        for (slat, slon), d in zip(school_coords, distances)
    ]
    avg_residual = np.mean(residuals)
    if avg_residual > 0.5:
        print(f"    Trilateration quality warning: avg residual = {avg_residual:.3f} mi")

    return round(est_lat, 6), round(est_lon, 6)

def geocode_via_trilateration(address, df):
    """
    Method 3: Geocode a home by trilaterating from its assigned school locations.
    Looks up school data from the original df using the home address.

    Parameters
    ----------
    address : str — the home address to geocode
    df      : original homes DataFrame containing school address/distance columns
    """
    school_fields = [
        ("elementary_school_address", "elementary_school_distance"),
        ("middle_school_address",     "middle_school_distance"),
        ("high_school_address",       "high_school_distance"),
    ]

    # Look up this address in the original df to get school data
    matching_rows = df[df["address"] == address]
    if matching_rows.empty:
        print(f"    Address not found in original df: {address}")
        return None, None

    row = matching_rows.iloc[0]
    school_coords = []
    distances     = []

    for addr_col, dist_col in school_fields:
        school_address  = row.get(addr_col)
        school_distance = row.get(dist_col)

        if pd.isna(school_address) or pd.isna(school_distance):
            print(f"    Missing school data in column '{addr_col}' — skipping trilateration")
            return None, None

        slat, slon = geocode_census(school_address)
        time.sleep(0.3)

        if slat is None:
            print(f"    School Census failed, trying Nominatim: {school_address}")
            slat, slon = geocode_nominatim(school_address)
            time.sleep(1)

        if slat is None:
            print(f"    Could not geocode school: {school_address}")
            return None, None

        school_coords.append((slat, slon))
        distances.append(float(school_distance))

    return trilaterate(school_coords, distances)


# ════════════════════════════════════════════════════════════════════════════
# MAIN HOME GEOCODING PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def geocode_all_homes(optw_input_df, df):
    """
    Geocode all homes in optw_input_df using a three-tier fallback pipeline:

      Tier 1 — Census geocoder       (free, no key, fast)
      Tier 2 — Nominatim / OSM       (free, no key, catches newer streets)
      Tier 3 — Trilateration         (uses school data from original df)

    Skips depot rows (interest_score == 0).
    Skips rows that already have lat/lon populated.
    Geocodes duplicate addresses only once — reuses result for identical addresses.

    Parameters
    ----------
    optw_input_df : filtered homes DataFrame (output of add_start_and_end)
    df            : original full homes DataFrame (used for school data lookup
                    in trilateration fallback only)

    Returns optw_input_df with latitude, longitude, geocode_method columns filled.
    """
    for col in ['latitude', 'longitude', 'geocode_method']:
        if col not in optw_input_df.columns:
            optw_input_df[col] = None

    # ── Cache: avoid re-geocoding the same address twice (e.g. same depot) ────
    geocode_cache = {}

    for idx, row in optw_input_df.iterrows():

        # Skip depot rows — geocoded separately in build_optw_coords
        if row['interest_score'] == 0:
            continue

        address = row["address"]

        # Skip if already geocoded
        if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
            print(f"⏭️  Already done: {address}")
            geocode_cache[address] = (row["latitude"], row["longitude"],
                                      row.get("geocode_method", "prior"))
            continue

        # Reuse cached result if this address already appeared earlier
        if address in geocode_cache:
            lat, lon, method = geocode_cache[address]
            optw_input_df.at[idx, "latitude"]       = lat
            optw_input_df.at[idx, "longitude"]      = lon
            optw_input_df.at[idx, "geocode_method"] = method + "_cached"
            print(f"♻️  Reused cache: {address} → {lat}, {lon}")
            continue

        lat, lon, method = None, None, None

        # ── Tier 1: Census ───────────────────────────────────────────────────
        print(f"🔍 [{idx}/{len(optw_input_df)-1}] Trying Census:      {address}")
        lat, lon = geocode_census(address)
        time.sleep(0.3)
        if lat is not None:
            method = "census"

        # ── Tier 2: Nominatim ────────────────────────────────────────────────
        if lat is None:
            print(f"   ⚠️  Census failed. Trying Nominatim: {address}")
            lat, lon = geocode_nominatim(address)
            time.sleep(1)
            if lat is not None:
                method = "nominatim"

        # ── Tier 3: Trilateration ────────────────────────────────────────────
        if lat is None:
            print(f"   ⚠️  Nominatim failed. Trying trilateration: {address}")
            lat, lon = geocode_via_trilateration(address, df)
            if lat is not None:
                method = "trilateration"

        # ── Record result ────────────────────────────────────────────────────
        if lat is not None:
            optw_input_df.at[idx, "latitude"]       = lat
            optw_input_df.at[idx, "longitude"]      = lon
            optw_input_df.at[idx, "geocode_method"] = method
            geocode_cache[address]                  = (lat, lon, method)
            print(f"   ✅ {method.upper()}: {lat}, {lon}")
        else:
            optw_input_df.at[idx, "geocode_method"] = "failed"
            print(f"   ❌ All methods failed for: {address}")

    return optw_input_df


def build_optw_coords(optw_input_df):
    """
    Build the ordered coordinate list for the duration matrix.
    Reads depot addresses from first and last rows of optw_input_df.
    Reads home coordinates from latitude/longitude columns in optw_input_df.
    Geocodes start depot once and reuses coordinates if start == end.

    Returns
    -------
    all_coords : list of (lat, lon) in optw_input_df row order
    start_idx  : index of start depot (always 0)
    end_idx    : index of end depot   (always len - 1)
    """
    start_address = optw_input_df['address'].iloc[0]
    end_address   = optw_input_df['address'].iloc[-1]

    # ── Geocode start depot ───────────────────────────────────────────────────
    print(f"📍 Geocoding start depot: {start_address}")
    start_lat, start_lon = geocode_depot(start_address)
    print(f"   ✅ {start_lat}, {start_lon}")

    # ── Reuse start coords if same address ────────────────────────────────────
    if end_address == start_address:
        end_lat, end_lon = start_lat, start_lon
        print(f"ℹ️  Start and end are the same — reusing coordinates.")
    else:
        print(f"📍 Geocoding end depot: {end_address}")
        end_lat, end_lon = geocode_depot(end_address)
        print(f"   ✅ {end_lat}, {end_lon}")

    # ── Home coordinates from optw_input_df ──────────────────────────────────
    home_rows   = optw_input_df.iloc[1:-1]
    home_coords = list(zip(home_rows['latitude'], home_rows['longitude']))

    all_coords = [(start_lat, start_lon)] + home_coords + [(end_lat, end_lon)]
    start_idx  = 0
    end_idx    = len(all_coords) - 1

    print(f"\n✅ Coordinate list built: {len(all_coords)} nodes "
          f"(1 start + {len(home_coords)} homes + 1 end)")

    return all_coords, start_idx, end_idx

# ════════════════════════════════════════════════════════════════════════════
# GEOCODING SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def geocode_summary(optw_input_df):
    """Print a summary of geocoding results for optw_input_df homes only."""
    homes = optw_input_df[optw_input_df['interest_score'] != 0]

    print("\n── Geocoding summary ───────────────────────────────────────────────")
    counts = homes["geocode_method"].value_counts()
    for method, count in counts.items():
        icon = {
            "census":            "✅",
            "nominatim":         "🌍",
            "trilateration":     "📐",
            "census_cached":     "✅",
            "nominatim_cached":  "🌍",
            "failed":            "❌",
        }.get(method, "❓")
        print(f"  {icon} {method:<20} {count} home(s)")

    failed = homes[homes["geocode_method"] == "failed"]
    if not failed.empty:
        print(f"\n  ⚠️  The following addresses need manual lat/lon:")
        for _, row in failed.iterrows():
            print(f"     - {row['address']}")
        print("  Look these up on Google Maps: right-click the pin → 'What's here?'")
        print("  Then patch them in manually:")
        print("    optw_input_df.loc[optw_input_df['address'] == '<address>', 'latitude']  = <lat>")
        print("    optw_input_df.loc[optw_input_df['address'] == '<address>', 'longitude'] = <lon>")
    else:
        print("\n  ✅ All homes successfully geocoded.")

    print("\n── Results ─────────────────────────────────────────────────────────")
    print(homes[["address", "latitude", "longitude", "geocode_method"]].to_string(index=False))