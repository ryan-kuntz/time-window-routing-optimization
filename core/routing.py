import requests
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from geopy.geocoders import Nominatim

from core.geocoding import haversine

# ════════════════════════════════════════════════════════════════════════════
# DURATION MATRIX
# ════════════════════════════════════════════════════════════════════════════

def build_duration_matrix_osrm(coords):
    coord_str = ";".join([f"{lon},{lat}" for lat, lon in coords])
    url       = f"http://router.project-osrm.org/table/v1/driving/{coord_str}"
    r         = requests.get(url, timeout=10).json()
    if r["code"] == "Ok":
        return [[round(t / 60, 1) for t in row] for row in r["durations"]], "osrm"
    raise ValueError(f"OSRM returned error code: {r['code']}")


def build_duration_matrix_haversine(coords, avg_speed_mph=25, circuity=1.35):
    n      = len(coords)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            dist    = haversine(*coords[i], *coords[j])
            minutes = (dist * circuity / avg_speed_mph) * 60
            row.append(round(minutes, 1))
        matrix.append(row)
    return matrix, "haversine"


def build_duration_matrix_from_coords(all_coords, start_idx, end_idx):
    """
    Build N×N driving duration matrix (minutes).
    Tries OSRM first, falls back to haversine.

    Same depot:
        Blocks impossible arcs (leaving end depot, arriving at start depot)
        so OR-Tools never routes through them. Matrix stays N×N.

    Different depot:
        Blocks direct depot↔depot arc so solver must visit at least one home.
    """
    BIG        = 99999
    same_depot = (all_coords[start_idx] == all_coords[end_idx])
    n          = len(all_coords)

    try:
        matrix, method = build_duration_matrix_osrm(all_coords)
        print(f"✅ Duration matrix built via OSRM ({n}×{n}).")
    except Exception as e:
        print(f"⚠️  OSRM failed ({e}). Falling back to haversine.")
        matrix, method = build_duration_matrix_haversine(all_coords)
        print(f"✅ Duration matrix built via haversine ({n}×{n}).")

    if same_depot:
        # Block all arcs leaving the end depot and arriving at the start depot
        # so OR-Tools never treats them as valid mid-route moves.
        # start_depot → end_depot set to 0 (same physical location, zero cost).
        for j in range(n):
            matrix[end_idx][j]   = BIG   # can't leave end depot
            matrix[j][start_idx] = BIG   # can't arrive at start depot mid-route
        matrix[start_idx][end_idx] = 0   # depot to itself = zero travel cost
        matrix[end_idx][start_idx] = BIG # end → start still forbidden
        print(f"ℹ️  Same depot — impossible arcs set to {BIG}. Matrix stays {n}×{n}.")
    else:
        # Block direct start→end and end→start arcs so solver
        # must visit at least one home before returning to end depot.
        matrix[start_idx][end_idx] = BIG
        matrix[end_idx][start_idx] = BIG
        print(f"ℹ️  Different depots — direct depot↔depot travel disabled.")

    print(f"   Method: {method} | Nodes: {n} | Same depot: {same_depot}")
    return matrix, method