"use client";

import { useEffect, useMemo, useState } from "react";
import { SiteHeader } from "@/components/site-header";
import { getProjection } from "@/lib/api";
import type { ProjectionPoint } from "@/lib/types";
import { Skeleton } from "@/components/ui/skeleton";

export default function MapPage() {
  const [points, setPoints] = useState<ProjectionPoint[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<ProjectionPoint | null>(null);

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

  return (
    <div className="flex min-h-full flex-1 flex-col">
      <SiteHeader compact />
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">
        <h1 className="text-3xl" style={{ fontFamily: "var(--font-serif)" }}>
          Style map
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
          Every outfield player-season, projected to 2-D. Neighbours here are close in the same
          metric the ranking uses. Click a dot to open that season.
        </p>
        {error && <p className="mt-4 text-sm text-destructive">{error}</p>}
        {!points && !error && <Skeleton className="mt-6 h-[480px] w-full" />}
        {points && layout && (
          <div className="relative mt-6 overflow-hidden rounded-2xl border border-border bg-card">
            <svg viewBox="0 0 1000 640" className="h-[min(70vh,640px)] w-full">
              {points.map((p) => {
                const x = ((p.x - layout.minX) / layout.w) * 960 + 20;
                const y = 620 - ((p.y - layout.minY) / layout.h) * 600;
                const highlighted = hover?.id === p.id;
                return (
                  <a key={p.id} href={`/player/${p.id}`}>
                    <circle
                      cx={x}
                      cy={y}
                      r={highlighted ? 5 : 2.2}
                      fill={p.primary_pos === "FW" ? "#e8c547" : p.primary_pos === "DF" ? "#7eb8da" : "#b6e388"}
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
            {hover && (
              <div className="pointer-events-none absolute bottom-3 left-3 rounded-lg bg-background/90 px-3 py-2 text-sm">
                <div className="font-medium">{hover.player}</div>
                <div className="text-muted-foreground">
                  {hover.season} · {hover.squad} · {hover.pos}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
