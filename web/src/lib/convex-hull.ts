export type Pt = { x: number; y: number };

function cross(o: Pt, a: Pt, b: Pt): number {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) return (sorted[mid - 1] + sorted[mid]) / 2;
  return sorted[mid];
}

export function convexHull(points: Pt[]): Pt[] {
  const uniq = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  if (uniq.length <= 2) return uniq;
  const lower: Pt[] = [];
  for (const p of uniq) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper: Pt[] = [];
  for (let i = uniq.length - 1; i >= 0; i--) {
    const p = uniq[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

/** Convex hull of the densest 80% of points so outliers do not swallow the map. */
export function compactHull(points: Pt[], keep = 0.8, pad = 10): Pt[] {
  if (points.length < 3) return [];
  const cx = median(points.map((p) => p.x));
  const cy = median(points.map((p) => p.y));
  const ranked = points
    .map((p) => ({ p, d: (p.x - cx) ** 2 + (p.y - cy) ** 2 }))
    .sort((a, b) => a.d - b.d);
  const cut = ranked.slice(0, Math.max(3, Math.ceil(ranked.length * keep))).map((row) => row.p);
  const hull = convexHull(cut);
  if (hull.length < 3) return [];
  const hx = hull.reduce((s, p) => s + p.x, 0) / hull.length;
  const hy = hull.reduce((s, p) => s + p.y, 0) / hull.length;
  return hull.map((p) => {
    const dx = p.x - hx;
    const dy = p.y - hy;
    const len = Math.hypot(dx, dy) || 1;
    return { x: p.x + (dx / len) * pad, y: p.y + (dy / len) * pad };
  });
}
