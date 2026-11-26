"""Line-crossing logic (pure Python, no ML dependencies)."""


def side_of_line(p, a, b):
    """Sign of which side of line a->b point p is on. >0, <0 or 0."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _within_segment(p, a, b, margin=0.1):
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy
    if denom == 0:
        return False
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / denom
    return -margin <= t <= 1 + margin


class LineCounter:
    def __init__(self, a, b, entry_direction="positive"):
        self.a, self.b = a, b
        self.entry_sign = 1 if entry_direction == "positive" else -1

    def check(self, prev_pt, cur_pt):
        """Return 'entry', 'exit' or None for a move prev_pt -> cur_pt."""
        if prev_pt is None:
            return None
        s0 = side_of_line(prev_pt, self.a, self.b)
        s1 = side_of_line(cur_pt, self.a, self.b)
        if s0 == 0 or s1 == 0 or (s0 > 0) == (s1 > 0):
            return None
        if not _within_segment(cur_pt, self.a, self.b):
            return None
        moved_sign = 1 if s1 > 0 else -1   # side we ended up on
        return "entry" if moved_sign == self.entry_sign else "exit"
