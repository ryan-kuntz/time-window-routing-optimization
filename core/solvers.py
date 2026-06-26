from itertools import permutations

from core.utils import time_to_minutes

def brute_force_optw(optw_input_df, duration_matrix):
    """
    Brute force solver for the Orienteering Problem with Time Windows (OPTW).

    Tries every possible ordering of home nodes, checks feasibility against
    hard time window constraints, and returns the route that maximizes total
    collected interest score.

    Parameters
    ----------
    optw_input_df   : DataFrame — rows 0 and -1 are depots, rows 1 to N-1 are homes
    duration_matrix : list[list[float]] — travel times in minutes (float)

    Returns
    -------
    dict with:
        best_route        — ordered list of addresses visited
        best_score        — total interest score collected
        best_schedule     — list of (address, arrival, departure) per stop
        total_travel_time — total driving time in minutes
        feasible_count    — number of feasible routes evaluated
        total_count       — total routes evaluated
    """
    # ── Parse nodes ──────────────────────────────────────────────────────────
    start_idx  = 0
    end_idx    = len(optw_input_df) - 1
    home_idxs  = list(range(1, end_idx))   # all rows between depots

    depot_start = optw_input_df.iloc[start_idx]
    depot_end   = optw_input_df.iloc[end_idx]

    tour_start_min = time_to_minutes(depot_start['open_house_start'])
    tour_end_min   = time_to_minutes(depot_end['open_house_end'])

    # Precompute time windows and scores for home nodes
    nodes = {}
    for i in home_idxs:
        row = optw_input_df.iloc[i]
        nodes[i] = {
            'address':         row['address'],
            'interest_score':  row['interest_score'],
            'tour_time':       row['tour_time'],
            'window_open':     time_to_minutes(row['open_house_start']),
            'window_close':    time_to_minutes(row['open_house_end']),
        }

    # ── Brute force ──────────────────────────────────────────────────────────
    best_score    = -1
    best_route    = None
    best_schedule = None
    best_travel   = None
    feasible_count = 0
    total_count    = 0

    # Try every subset and every ordering of home nodes
    from itertools import combinations

    for r in range(1, len(home_idxs) + 1):
        for subset in combinations(home_idxs, r):
            for perm in permutations(subset):
                total_count += 1
                route        = list(perm)
                current_time = tour_start_min
                total_score  = 0
                total_travel = 0
                schedule     = []
                feasible     = True

                prev_idx = start_idx

                for node_idx in route:
                    node     = nodes[node_idx]
                    travel   = duration_matrix[prev_idx][node_idx]
                    arrive   = current_time + travel
                    total_travel += travel

                    # Wait if arriving before open house starts
                    visit_start = max(arrive, node['window_open'])

                    # Hard constraint: must arrive before open house closes
                    # and must have enough time for the full visit
                    if arrive > node['window_close']:
                        feasible = False
                        break
                    if visit_start + node['tour_time'] > node['window_close']:
                        feasible = False
                        break

                    depart       = visit_start + node['tour_time']
                    current_time = depart

                    # Hard constraint: must be able to return to end depot in time
                    time_to_end = duration_matrix[node_idx][end_idx]
                    if current_time + time_to_end > tour_end_min:
                        feasible = False
                        break

                    total_score += node['interest_score']
                    schedule.append({
                        'address':    node['address'],
                        'arrival':    f"{int(arrive // 60):02d}:{int(arrive % 60):02d}",
                        'wait':       round(max(0, node['window_open'] - arrive), 1),
                        'visit_start':f"{int(visit_start // 60):02d}:{int(visit_start % 60):02d}",
                        'departure':  f"{int(depart // 60):02d}:{int(depart % 60):02d}",
                        'score':      node['interest_score'],
                    })
                    prev_idx = node_idx

                if feasible and total_score > best_score:
                    best_score    = total_score
                    best_route    = [depot_start['address']] \
                                  + [nodes[i]['address'] for i in route] \
                                  + [depot_end['address']]
                    best_schedule = schedule
                    best_travel   = total_travel
                    feasible_count += 1

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n── Brute force OPTW results ─────────────────────────────────────────")
    print(f"   Routes evaluated:  {total_count:,}")
    print(f"   Feasible routes:   {feasible_count:,}")
    print(f"   Best score:        {best_score:.1f}")
    print(f"   Total drive time:  {best_travel:.1f} min")
    print(f"\n── Optimal route ────────────────────────────────────────────────────")
    for i, stop in enumerate(best_schedule):
        print(f"   {i+1}. {stop['address']}")
        print(f"      Arrive: {stop['arrival']}  "
              f"{'(wait ' + str(stop['wait']) + ' min)  ' if stop['wait'] > 0 else ''}"
              f"Visit: {stop['visit_start']}  Depart: {stop['departure']}  "
              f"Score: {stop['score']:.1f}")
    print(f"\n   🏁 Return to depot by: "
          f"{int((time_to_minutes(depot_end['open_house_end'])) // 60):02d}:"
          f"{int((time_to_minutes(depot_end['open_house_end'])) % 60):02d}")

    return {
        'best_route':        best_route,
        'best_score':        best_score,
        'best_schedule':     best_schedule,
        'total_travel_time': best_travel,
        'feasible_count':    feasible_count,
        'total_count':       total_count,
    }

def greedy_optw(optw_input_df, duration_matrix):
    """
    Greedy heuristic for OPTW.

    At each step, selects the unvisited home with the highest interest score
    that is still feasible given the current time and remaining tour window.
    Does NOT backtrack — once a home is skipped it is never reconsidered.

    Parameters
    ----------
    optw_input_df   : DataFrame — row 0 = start depot, rows 1 to N-1 = homes,
                      row N = end depot
    duration_matrix : list[list[float]] — travel times in minutes

    Returns
    -------
    dict with best_route, best_score, best_schedule, total_travel_time
    """
    import time as time_module
    solve_start = time_module.time()

    start_idx = 0
    end_idx   = len(optw_input_df) - 1

    depot_start    = optw_input_df.iloc[start_idx]
    depot_end      = optw_input_df.iloc[end_idx]
    tour_start_min = time_to_minutes(depot_start['open_house_start'])
    tour_end_min   = time_to_minutes(depot_end['open_house_end'])

    # ── Build candidate pool ──────────────────────────────────────────────────
    candidates = {}
    for i in range(1, end_idx):
        row = optw_input_df.iloc[i]
        candidates[i] = {
            'address':        row['address'],
            'interest_score': row['interest_score'],
            'tour_time':      row['tour_time'],
            'window_open':    time_to_minutes(row['open_house_start']),
            'window_close':   time_to_minutes(row['open_house_end']),
        }

    # ── Greedy selection loop ─────────────────────────────────────────────────
    visited      = []
    schedule     = []
    current_time = tour_start_min
    current_idx  = start_idx
    total_score  = 0
    total_travel = 0
    remaining    = set(candidates.keys())

    while remaining:
        # Sort unvisited homes by interest score descending
        # (greedy criterion: always try highest score first)
        sorted_candidates = sorted(
            remaining,
            key=lambda i: candidates[i]['interest_score'],
            reverse=True
        )

        selected = None
        for idx in sorted_candidates:
            node   = candidates[idx]
            travel = duration_matrix[current_idx][idx]
            arrive = current_time + travel

            # Hard constraint 1: must arrive before open house closes
            if arrive > node['window_close']:
                continue

            # Hard constraint 2: must have enough time for full visit
            visit_start = max(arrive, node['window_open'])
            if visit_start + node['tour_time'] > node['window_close']:
                continue

            # Hard constraint 3: must be able to return to end depot in time
            depart      = visit_start + node['tour_time']
            time_to_end = duration_matrix[idx][end_idx]
            if depart + time_to_end > tour_end_min:
                continue

            selected = idx
            break   # take the first feasible home (highest score)

        if selected is None:
            break   # no feasible home remaining — stop

        # ── Commit to selected home ───────────────────────────────────────────
        node        = candidates[selected]
        travel      = duration_matrix[current_idx][selected]
        arrive      = current_time + travel
        visit_start = max(arrive, node['window_open'])
        depart      = visit_start + node['tour_time']

        total_travel += travel
        total_score  += node['interest_score']
        current_time  = depart
        current_idx   = selected

        schedule.append({
            'address':    node['address'],
            'arrival':    f"{int(arrive // 60):02d}:{int(arrive % 60):02d}",
            'wait':       round(max(0, node['window_open'] - arrive), 1),
            'visit_start':f"{int(visit_start // 60):02d}:{int(visit_start % 60):02d}",
            'departure':  f"{int(depart // 60):02d}:{int(depart % 60):02d}",
            'score':      node['interest_score'],
        })
        visited.append(selected)
        remaining.remove(selected)

    solve_time = time_module.time() - solve_start

    # ── Build route list ──────────────────────────────────────────────────────
    best_route = (
        [depot_start['address']]
        + [candidates[i]['address'] for i in visited]
        + [depot_end['address']]
    )

    # ── Print results ─────────────────────────────────────────────────────────
    max_possible = sum(c['interest_score'] for c in candidates.values())

    print(f"\n── Greedy OPTW results ──────────────────────────────────────────────")
    print(f"   Homes visited:     {len(visited)}/{len(candidates)}")
    print(f"   Interest score:    {total_score:.1f} / {max_possible:.1f} "
          f"({100 * total_score / max_possible:.1f}% of max)")
    print(f"   Total drive time:  {total_travel:.1f} min")
    print(f"   Solve time:        {solve_time:.4f}s")
    print(f"\n── Route ────────────────────────────────────────────────────────────")
    print(f"   Start: {depot_start['address']}  {depot_start['open_house_start']}")
    for i, stop in enumerate(schedule):
        wait_str = f"(wait {stop['wait']} min)  " if stop['wait'] > 0 else ""
        print(f"   {i+1}. {stop['address']}")
        print(f"      Arrive: {stop['arrival']}  {wait_str}"
              f"Visit: {stop['visit_start']}  Depart: {stop['departure']}  "
              f"Score: {stop['score']:.1f}")
    print(f"   End:   {depot_end['address']}")

    return {
        'method':            'greedy',
        'best_route':        best_route,
        'best_score':        total_score,
        'best_schedule':     schedule,
        'total_travel_time': total_travel,
        'solve_time':        solve_time,
        'homes_visited':     len(visited),
        'total_homes':       len(candidates),
        'max_possible':      max_possible,
    }

