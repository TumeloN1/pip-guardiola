"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { SiteHeader } from "@/components/site-header";
import { MapFilters } from "@/components/map-filters";
import { getProjection } from "@/lib/api";
import { defaultMapFilters, pointMatchesMapFilter, type MapFilterState } from "@/lib/map-filters";
import type { ProjectionPoint } from "@/lib/types";
import { Skeleton } from "@/components/ui/skeleton";

export default function MapPage() {
  const [points, setPoints] = useState<ProjectionPoint[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<ProjectionPoint | null>(null);
  const [filters, setFilters] = useState<MapFilterState>(defaultMapFilters);

  useEffect(() => {
    getProjection("outfield")
      .then((res) => setPoints(res.points))
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

  return (
    <div className="flex min-h-full flex-1 flex-col">
      <SiteHeader />
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">
        <h1 className="font-heading text-4xl uppercase tracking-tight text-primary">
          Style map
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
          Every outfield player-season, projected to 2-D. Neighbours here are close in the same
          metric the ranking uses. Filters hide dots but keep the axes fixed, so you can see where
          a league or role sits in the space. Click a dot to open that season.
        </p>
        <p className="mt-2 text-sm">
          <Link href="/map/concepts" className="font-semibold text-primary hover:text-[#FF2882]">
            Three ways to label neighbourhoods
          </Link>
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
            <div className="relative mt-4 overflow-hidden border border-border bg-card">
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
                  {visible.map((p) => {
                    const x = ((p.x - layout.minX) / layout.w) * 960 + 20;
                    const y = 620 - ((p.y - layout.minY) / layout.h) * 600;
                    const highlighted = hover?.id === p.id;
                    return (
                      <a key={p.id} href={`/player/${p.id}`}>
                        <circle
                          cx={x}
                          cy={y}
                          r={highlighted ? 5 : 2.2}
                          fill={
                            p.primary_pos === "FW"
                              ? "#FF2882"
                              : p.primary_pos === "DF"
                                ? "#37003C"
                                : "#00C853"
                          }
                          opacity={highlighted ? 1 : 0.7}
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
                </div>
              )}
            </div>
            <div className="mt-3 flex flex-wrap gap-4 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">
              <span className="inline-flex items-center gap-2">
                <span className="size-2.5 bg-[#37003C]" /> DF
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="size-2.5 bg-[#00C853]" /> MF
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="size-2.5 bg-[#FF2882]" /> FW
              </span>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
