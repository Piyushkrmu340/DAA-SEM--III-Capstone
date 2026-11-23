"""
Automated Examination Timetable Generator
=========================================
Model
-----
  * Each SUBJECT  -> a vertex.
  * Two subjects that share at least one student -> an edge (a conflict).
  * Each EXAM SLOT -> a colour.
  * A valid timetable = a proper vertex colouring (no edge joins two equal colours).
  * Minimum number of slots = chromatic number chi(G).

Algorithm
---------
  m-colouring decision problem solved by BACKTRACKING, wrapped in a search
  that tries m = lower_bound, lower_bound+1, ... until a colouring exists.
  The first m that succeeds is therefore the minimum (chromatic number).

  Backtracking enhancements (they keep it exact, only faster):
    1. Lower bound  : size of a greedy clique (a clique of size q needs q slots).
    2. Upper bound  : DSATUR greedy colouring (guarantees termination at m<=ub).
    3. MRV ordering : always branch on the vertex with the fewest legal colours
                      (ties -> highest degree).
    4. Forward check: if any uncoloured vertex has zero legal colours -> prune.
    5. Symmetry cut : a vertex may only open the *next unused* colour, so
                      permutations of identical colourings are not re-explored.

Usage
-----
  python exam_timetable.py                      # built-in demo
  python exam_timetable.py enrol.csv            # CSV with columns: student,subject
  python exam_timetable.py enrol.json           # JSON {"student": ["Sub1","Sub2"], ...}
  python exam_timetable.py enrol.csv --start 2026-12-01 --slots-per-day 2
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import date, timedelta


# --------------------------------------------------------------------------- #
# 1. Conflict graph
# --------------------------------------------------------------------------- #
class ConflictGraph:
    """Undirected graph: vertices = subjects, edges = shared students."""

    def __init__(self):
        self.adj = {}                       # subject -> set(conflicting subjects)
        self.shared = defaultdict(set)      # (u,v) sorted -> set(students)

    def add_vertex(self, v):
        self.adj.setdefault(v, set())

    def add_edge(self, u, v, student=None):
        if u == v:
            return
        self.add_vertex(u)
        self.add_vertex(v)
        self.adj[u].add(v)
        self.adj[v].add(u)
        if student is not None:
            self.shared[tuple(sorted((u, v)))].add(student)

    @classmethod
    def from_enrollments(cls, enrollments):
        """enrollments: {student: iterable of subjects}"""
        g = cls()
        for student, subjects in enrollments.items():
            subjects = sorted(set(subjects))
            for s in subjects:
                g.add_vertex(s)
            for i in range(len(subjects)):
                for j in range(i + 1, len(subjects)):
                    g.add_edge(subjects[i], subjects[j], student)
        return g

    @property
    def vertices(self):
        return sorted(self.adj)

    def edges(self):
        return sorted({tuple(sorted((u, v))) for u in self.adj for v in self.adj[u]})

    def degree(self, v):
        return len(self.adj[v])


# --------------------------------------------------------------------------- #
# 2. Bounds
# --------------------------------------------------------------------------- #
def greedy_clique_size(g):
    """Lower bound on chi(G): size of a (greedily found) clique."""
    best = 1 if g.adj else 0
    for start in g.vertices:
        clique = [start]
        cands = set(g.adj[start])
        while cands:
            v = max(cands, key=lambda x: len(g.adj[x] & cands))
            clique.append(v)
            cands &= g.adj[v]
        best = max(best, len(clique))
    return best


def dsatur_coloring(g):
    """Fast greedy colouring -> an upper bound on chi(G)."""
    color = {}
    while len(color) < len(g.adj):
        def key(v):
            sat = len({color[n] for n in g.adj[v] if n in color})
            return (sat, g.degree(v))
        v = max((x for x in g.vertices if x not in color), key=key)
        used = {color[n] for n in g.adj[v] if n in color}
        c = 0
        while c in used:
            c += 1
        color[v] = c
    return color


# --------------------------------------------------------------------------- #
# 3. Backtracking m-colouring
# --------------------------------------------------------------------------- #
class BacktrackingColoring:
    def __init__(self, g):
        self.g = g
        self.nodes_explored = 0

    def color_with(self, m):
        """Try to colour the graph with at most m colours. Returns dict or None."""
        g, color = self.g, {}
        n = len(g.adj)

        def legal(v):
            banned = {color[x] for x in g.adj[v] if x in color}
            return [c for c in range(m) if c not in banned]

        def solve(used):
            if len(color) == n:
                return True
            # MRV selection + forward checking
            best_v, best_opts = None, None
            for v in g.vertices:
                if v in color:
                    continue
                opts = legal(v)
                if not opts:
                    return False                       # dead end -> prune
                if (best_v is None
                        or len(opts) < len(best_opts)
                        or (len(opts) == len(best_opts)
                            and g.degree(v) > g.degree(best_v))):
                    best_v, best_opts = v, opts
            for c in best_opts:
                if c > used:                           # symmetry breaking
                    break
                self.nodes_explored += 1
                color[best_v] = c
                if solve(max(used, c + 1)):
                    return True
                del color[best_v]                      # BACKTRACK
            return False

        return dict(color) if solve(0) else None

    def minimum_coloring(self):
        """Smallest m for which a proper colouring exists = chromatic number."""
        if not self.g.adj:
            return 0, {}
        lb = greedy_clique_size(self.g)
        ub_col = dsatur_coloring(self.g)
        ub = max(ub_col.values()) + 1
        for m in range(lb, ub):
            sol = self.color_with(m)
            if sol is not None:
                return m, sol
        return ub, ub_col          # lb..ub-1 all failed -> DSATUR is optimal


# --------------------------------------------------------------------------- #
# 4. Verification & timetable building
# --------------------------------------------------------------------------- #
def verify(g, coloring, enrollments):
    """Independent check: no student has two exams in one slot."""
    assert set(coloring) == set(g.adj), "some subject is unscheduled"
    for u, v in g.edges():
        assert coloring[u] != coloring[v], f"conflict: {u} & {v}"
    for student, subs in enrollments.items():
        slots = [coloring[s] for s in set(subs)]
        assert len(slots) == len(set(slots)), f"clash for student {student}"
    return True


SLOT_TIMES = ["09:00-12:00", "14:00-17:00", "18:00-21:00", "07:00-09:00"]


def calendar_for_slots(n_slots, start, per_day, skip_weekends=True):
    """Map slot index -> (date, time-of-day)."""
    out, d, k = [], start, 0
    while len(out) < n_slots:
        if skip_weekends and d.weekday() >= 5:
            d += timedelta(days=1)
            continue
        out.append((d, SLOT_TIMES[k % len(SLOT_TIMES)]))
        k += 1
        if k == per_day:
            k, d = 0, d + timedelta(days=1)
    return out


def print_report(g, m, coloring, nodes, secs, enrollments, start, per_day):
    print("=" * 66)
    print(" EXAMINATION TIMETABLE (Graph Colouring + Backtracking)")
    print("=" * 66)
    print(f" Subjects (vertices) : {len(g.adj)}")
    print(f" Conflicts (edges)   : {len(g.edges())}")
    print(f" Students            : {len(enrollments)}")
    print(f" Minimum slots (chi) : {m}")
    print(f" Search nodes        : {nodes}   Time: {secs*1000:.2f} ms")
    print("-" * 66)

    groups = defaultdict(list)
    for s, c in coloring.items():
        groups[c].append(s)
    cal = calendar_for_slots(m, start, per_day)
    for c in range(m):
        d, t = cal[c]
        print(f" Slot {c+1:>2} | {d:%a %d %b %Y} {t} | "
              + ", ".join(sorted(groups[c])))
    print("-" * 66)
    print(" Conflict graph (subject -> clashes with):")
    for v in g.vertices:
        print(f"   {v:<12} -> {', '.join(sorted(g.adj[v])) or '-'}")
    print("-" * 66)
    print(" Verification: PASSED - no student has two exams in the same slot.")
    print("=" * 66)


# --------------------------------------------------------------------------- #
# 5. Input
# --------------------------------------------------------------------------- #
def load_enrollments(path):
    enr = defaultdict(set)
    if path.lower().endswith(".json"):
        with open(path) as f:
            for stu, subs in json.load(f).items():
                enr[stu].update(subs)
    else:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                enr[row["student"].strip()].add(row["subject"].strip())
    return enr


DEMO = {
    "S01": ["Maths", "Physics", "Chemistry"],
    "S02": ["Maths", "English", "CS"],
    "S03": ["Physics", "CS", "Biology"],
    "S04": ["Chemistry", "Biology", "English"],
    "S05": ["Maths", "Economics"],
    "S06": ["Economics", "History", "English"],
    "S07": ["History", "Physics"],
    "S08": ["CS", "Economics"],
    "S09": ["Chemistry", "History"],
    "S10": ["Biology", "Maths"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="CSV (student,subject) or JSON enrolment file")
    ap.add_argument("--start", default="2026-12-01", help="first exam date, YYYY-MM-DD")
    ap.add_argument("--slots-per-day", type=int, default=2, choices=[1, 2, 3, 4])
    a = ap.parse_args()

    enrollments = load_enrollments(a.file) if a.file else {k: set(v) for k, v in DEMO.items()}
    g = ConflictGraph.from_enrollments(enrollments)

    solver = BacktrackingColoring(g)
    t0 = time.perf_counter()
    m, coloring = solver.minimum_coloring()
    dt = time.perf_counter() - t0

    verify(g, coloring, enrollments)
    print_report(g, m, coloring, solver.nodes_explored, dt, enrollments,
                 date.fromisoformat(a.start), a.slots_per_day)


if __name__ == "__main__":
    sys.exit(main())