"use client";

import { useEffect, useMemo, useState } from "react";
import { SiteHeader } from "@/components/site-header";
import { MapFilters } from "@/components/map-filters";
import { MapStyleRail } from "@/components/map-style-rail";
import { getProjection } from "@/lib/api";
import { compactHull } from "@/lib/convex-hull";
import { defaultMapFilters, pointMatchesMapFilter, type MapFilterState } from "@/lib/map-filters";
import { styleColor } from "@/lib/map-styles";
import type { ProjectionPoint } from "@/lib/types";
import { Skeleton } from "@/components/ui/skeleton";

type Layout = { minX: number; maxX: number; minY: number; maxY: number; w: number; h: number };

function project(p: ProjectionPoint, layout: Layout) {
  return {
    x: ((p.x - layout.minX) / layout.w) * 960 + 20,
    y: 620 - ((p.y - layout.minY) / layout.h) * 600,
  };
}

export default function MapPage() {
  const [points, setPoints] = useState<ProjectionPoint[] | null>(null);
  const [catalog, setCatalog] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<ProjectionPoint | null>(null);
  const [filters, setFilters] = useState<MapFilterState>(defaultMapFilters);
  const [selected, setSelected] = useState<string[]>([]);

  useEffect(() => {
    getProjection("outfield")
      .then((res) => {
        setPoints(res.points);
        const fromApi = res.styles ?? [];
        setCatalog(
          fromApi.length
            ? fromApi
            : [...new Set(res.points.map((p) => p.style).filter((name): name is string => Boolean(name)))],
        );
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed to load map"));
  }, []);

  const layout = useMemo(() => {
    if (!points || points.length === 0) return null;
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    return { minX, maxX, minY, maxY, w: maxX - minX || 1, h: maxY - minY || 1 };
  }, [points]);

  const visible = useMemo(() => {
    if (!points) return [];
    return points.filter((point) => pointMatchesMapFilter(point, filters));
  }, [points, filters]);

  const counts = useMemo(() => {
    const next: Record<string, number> = {};
    for (const point of visible) {
      if (!point.style) continue;
      next[point.style] = (next[point.style] ?? 0) + 1;
    }
    return next;
  }, [visible]);

  const hulls = useMemo(() => {
    if (!layout || selected.length === 0) return [];
    return selected
      .map((name) => {
        const pts = visible.filter((p) => p.style === name).map((p) => project(p, layout));
        return { name, color: styleColor(name).fill, hull: compactHull(pts) };
      })
      .filter((row) => row.hull.length >= 3);
  }, [layout, selected, visible]);

  function toggleStyle(name: string) {
    setSelected((cur) => (cur.includes(name) ? cur.filter((x) => x !== name) : [...cur, name]));
  }

  const isolating = selected.length > 0;

  return (
    <div className="flex min-h-full flex-1 flex-col">
      <SiteHeader />
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">
        <h1 className="font-heading text-4xl uppercase tracking-tight text-primary">
          Style map
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
          Every outfield player-season, projected to 2-D. Neighbours here are close in the same
          metric the ranking uses. Filters hide dots but keep the axes fixed. Pick a style to see
          where that neighbourhood sits — not a stat on either axis. Click a dot to open that season.
        </p>
        {error && <p className="mt-4 text-sm text-destructive">{error}</p>}
        {!points && !error && <Skeleton className="mt-6 h-[480px] w-full" />}
        {points && layout && (
          <>
            <MapFilters
              value={filters}
              onChange={setFilters}
              visible={visible.length}
              total={points.length}
            />
            <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_16rem]">
              <div className="order-2 lg:order-1">
                <div className="relative overflow-hidden border border-border bg-card">
                  {visible.length === 0 ? (
                    <div className="flex h-[min(70vh,640px)] items-center justify-center px-6 text-center text-sm text-muted-foreground">
                      No seasons match these filters. Reset or widen the era.
                    </div>
                  ) : (
                    <svg
                      viewBox="0 0 1000 640"
                      className="h-[min(70vh,640px)] w-full"
                      onMouseLeave={() => setHover(null)}
                    >
                      {hulls.map((row) => (
                        <polygon
                          key={row.name}
                          points={row.hull.map((p) => `${p.x},${p.y}`).join(" ")}
                          fill={row.color}
                          fillOpacity={0.1}
                          stroke={row.color}
                          strokeWidth={2}
                          strokeDasharray="8 6"
                          pointerEvents="none"
                        />
                      ))}
                      {visible.map((p) => {
                        const { x, y } = project(p, layout);
                        const highlighted = hover?.id === p.id;
                        const active = isolating && selected.includes(p.style ?? "");
                        const color = active
                          ? styleColor(p.style ?? "").fill
                          : isolating
                            ? "#37003C"
                            : "#5C4B60";
                        const opacity = highlighted ? 1 : isolating ? (active ? 0.95 : 0.08) : 0.28;
                        const r = highlighted ? 5.5 : active ? 3.4 : 2.2;
                        return (
                          <a key={p.id} href={`/player/${p.id}`}>
                            <circle
                              cx={x}
                              cy={y}
                              r={r}
                              fill={color}
                              opacity={opacity}
                              onMouseEnter={() => setHover(p)}
                            >
                              <title>
                                {p.player} · {p.season}
                              </title>
                            </circle>
                          </a>
                        );
                      })}
                    </svg>
                  )}
                  {hover && visible.length > 0 && (
                    <div className="pointer-events-none absolute bottom-3 left-3 right-3 border border-border bg-white px-3 py-2 text-sm shadow-sm sm:right-auto">
                      <div className="font-medium">{hover.player}</div>
                      <div className="text-muted-foreground">
                        {hover.season} · {hover.squad} · {hover.pos} · {hover.comp}
                      </div>
                      {hover.style && (
                        <div className="mt-1 text-xs font-semibold uppercase tracking-[0.14em] text-primary">
                          {hover.style}
                        </div>
                      )}
                    </div>
                  )}
                </div>
                {isolating && (
                  <div className="mt-3 flex flex-wrap gap-3 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                    {selected.map((name) => {
                      const n = counts[name] ?? 0;
                      const share = visible.length ? Math.round((n / visible.length) * 100) : 0;
                      return (
                        <span key={name} className="inline-flex items-center gap-2">
                          <span className="size-2.5" style={{ backgroundColor: styleColor(name).fill }} />
                          {name} · {share}% of this view
                        </span>
                      );
                    })}
                  </div>
                )}
              </div>
              <div className="order-1 lg:order-2">
                <MapStyleRail
                  catalog={catalog}
                  counts={counts}
                  selected={selected}
                  visible={visible.length}
                  onToggle={toggleStyle}
                  onClear={() => setSelected([])}
                />
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
