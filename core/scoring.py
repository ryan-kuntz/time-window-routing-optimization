import pandas as pd
import numpy as np

def normalize(series):
    """Min-max scale a Series to [0, 1]. Constant series → 0.5."""
    lo, hi = series.min(), series.max()
    return (series - lo) / (hi - lo) if hi > lo else pd.Series([0.5] * len(series), index=series.index)

def apply_dealbreakers(df, db):
    """Return a boolean mask — True = home survives all dealbreakers."""
    mask = pd.Series(True, index=df.index)
    if db['no_hoa']:
        mask &= (df['hoa_fee'] == 0)
    if db['min_beds']   > 0: mask &= (df['beds']          >= db['min_beds'])
    if db['min_baths']  > 0: mask &= (df['baths']         >= db['min_baths'])
    if db['min_sqft']   > 0: mask &= (df['sqft']          >= db['min_sqft'])
    if db['max_price']  > 0: mask &= (df['price']         <= db['max_price'])
    if db['min_garage'] > 0: mask &= (df['garage_spaces'] >= db['min_garage'])
    if db['min_year']   > 0: mask &= (df['year_built']    >= db['min_year'])
    return mask

def compute_neighborhood_score(df, vibe, vibe_weight):
    """
    Combine the five neighborhood scores into one composite.
    The preferred vibe column gets a 2x multiplier; others get 1x.
    All five are then averaged and returned as a normalized [0,1] series.
    """
    vibe_col_map = {
        'quiet':    'quiet_score',
        'walk':     'walk_score',
        'bike':     'bike_score',
        'wellness': 'wellness_score',
        'vibrancy': 'vibrancy_score',
    }
    cols    = ['quiet_score', 'walk_score', 'bike_score', 'wellness_score', 'vibrancy_score']
    weights = {c: 1.0 for c in cols}
    if vibe in vibe_col_map:
        weights[vibe_col_map[vibe]] = 2.0          # preferred vibe 2× bonus
    total_w  = sum(weights.values())
    composite = sum(normalize(df[c]) * (weights[c] / total_w) for c in cols)
    return composite  # already in [0, 1] because each normalize() → [0,1]


def score_homes(df, prefs):
    """
    Full scoring pipeline:
      1. Apply dealbreakers → survivors only
      2. Normalize each feature column
      3. Weighted sum → raw_score
      4. Rescale raw_score to 0–100
      5. Apply property-type penalty
      6. Return full df with interest_score column, sorted descending
    """
    db   = prefs['dealbreakers']
    w    = prefs['weights']
    vibe = prefs['vibe']
    vw   = prefs['vibe_weight']

    # Step 1 — dealbreaker filter
    surviving = df[apply_dealbreakers(df, db)].copy()
    n_removed = len(df) - len(surviving)
    print(f"Dealbreaker filter: {n_removed} home(s) removed, {len(surviving)} remaining.")

    if surviving.empty:
        print("⚠️  No homes survived dealbreakers — please relax your constraints.")
        return surviving

    # Step 2 & 3 — normalize features and apply weights
    # price and hoa_fee are inverted: lower value = better score
    surviving['_price_norm']    = 1 - normalize(surviving['price'])
    surviving['_sqft_norm']     = normalize(surviving['sqft'])
    surviving['_beds_norm']     = normalize(surviving['beds'])
    surviving['_baths_norm']    = normalize(surviving['baths'])
    surviving['_beds_baths']    = (surviving['_beds_norm'] + surviving['_baths_norm']) / 2
    surviving['_year_norm']     = normalize(surviving['year_built'])
    surviving['_lot_norm']      = normalize(surviving['lot_size_sqft'])
    surviving['_garage_norm']   = normalize(surviving['garage_spaces'])
    surviving['_hoa_norm']      = 1 - normalize(surviving['hoa_fee'])  # lower HOA = better
    surviving['_neigh_norm']    = compute_neighborhood_score(surviving, vibe, vw)

    # Map weight keys to computed normalized columns
    feature_map = {
        'price':      '_price_norm',
        'sqft':       '_sqft_norm',
        'beds_baths': '_beds_baths',
        'year_built': '_year_norm',
        'lot_size':   '_lot_norm',
        'garage':     '_garage_norm',
        'hoa_fee':    '_hoa_norm',
    }

    total_feature_weight = sum(w.values()) + vw
    raw = pd.Series(0.0, index=surviving.index)

    for key, col in feature_map.items():
        raw += surviving[col] * (w.get(key, 0) / total_feature_weight)

    # Neighborhood composite
    raw += surviving['_neigh_norm'] * (vw / total_feature_weight)

    # Step 4 — rescale to 0–100
    surviving['interest_score'] = (normalize(raw) * 100).round(1)

    # Step 5 — property type penalty (−10 pts if type not preferred)
    preferred = prefs['preferred_types']
    if 'No preference' not in preferred:
        penalty_mask = ~surviving['property_type'].isin(preferred)
        surviving.loc[penalty_mask, 'interest_score'] -= 10
        surviving['interest_score'] = surviving['interest_score'].clip(lower=0)

    # Drop internal computation columns
    surviving.drop(columns=[c for c in surviving.columns if c.startswith('_')], inplace=True)

    return surviving.sort_values('interest_score', ascending=False)