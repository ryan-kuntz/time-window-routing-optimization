def time_to_minutes(t):
    """Convert 'HH:MM' string to minutes from midnight."""
    h, m = map(int, t.split(":"))
    return h * 60 + m