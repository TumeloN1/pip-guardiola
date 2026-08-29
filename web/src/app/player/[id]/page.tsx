"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { SiteHeader } from "@/components/site-header";
import { RadarCard } from "@/components/radar-card";
import { FiltersCard, type FilterState } from "@/components/filters-card";
import { WeightSliders } from "@/components/weight-sliders";
import { SimilarTable } from "@/components/similar-table";
import { Badge } from "@/components/ui/badge";
import { getPlayer, getProfile, getSimilar } from "@/lib/api";
import type { PlayerHit, ProfileResponse, SimilarRow } from "@/lib/types";

const OUTFIELD_WEIGHTS = {
  finishing: 1,
  creation: 1,
  passing: 1,
  carrying: 1,
  occupation: 1,
  defending: 1,
  duels: 1,
};

const GK_WEIGHTS = {
  shotstopping: 1,
  distribution: 1,
  sweeping: 1,
};

export default function PlayerPage() {
  const params = useParams<{ id: string }>();
  const id = params.id;
  const [player, setPlayer] = useState<PlayerHit | null>(null);
  const [profile, setProfile] = useState<ProfileResponse | null>(null);
  const [rows, setRows] = useState<SimilarRow[] | null>(null);
  const [metric, setMetric] = useState("zscore_cosine");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [rankError, setRankError] = useState<string | null>(null);
  const [loadingPlayer, setLoadingPlayer] = useState(true);
  const [loadingRank, setLoadingRank] = useState(true);
  const [filters, setFilters] = useState<FilterState>({
    eraStart: 2018,
    eraEnd: 2025,
    comps: ["Premier League"],
    positions: [],
    minMinutes: 900,
  });
  const [weights, setWeights] = useState<Record<string, number>>(OUTFIELD_WEIGHTS);

  useEffect(() => {
    let cancelled = false;
    setLoadingPlayer(true);
    setLoadError(null);
    Promise.all([getPlayer(id), getProfile(id)])
      .then(([p, prof]) => {
        if (cancelled) return;
        setPlayer(p);
        setProfile(prof);
        setWeights(p.role === "keeper" ? GK_WEIGHTS : OUTFIELD_WEIGHTS);
        if (p.role === "keeper") {
          setFilters((f) => ({ ...f, positions: ["GK"], comps: [] }));
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) setLoadError(err instanceof Error ? err.message : "Failed to load player");
      })
      .finally(() => {
        if (!cancelled) setLoadingPlayer(false);
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  const similarKey = useMemo(
    () => JSON.stringify({ id, filters, weights }),
    [id, filters, weights],
  );

  useEffect(() => {
    let cancelled = false;
    setLoadingRank(true);
    setRankError(null);
    getSimilar(id, {
      eraStart: filters.eraStart,
      eraEnd: filters.eraEnd,
      comps: filters.comps,
      positions: filters.positions,
      minMinutes: filters.minMinutes,
      weights,
    })
      .then((res) => {
        if (cancelled) return;
        setRows(res.results);
        setMetric(res.metric);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setRankError(err instanceof Error ? err.message : "Ranking failed");
          setRows([]);
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingRank(false);
      });
    return () => {
      cancelled = true;
    };
  }, [similarKey, id, filters, weights]);

  return (
    <div className="flex min-h-full flex-1 flex-col">
      <SiteHeader compact />
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:py-8">
        {loadError && (
          <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm">
            {loadError.includes("unknown")
              ? "That player-season is not in the index. It may be below 900 minutes or missing advanced stats."
              : loadError}
          </div>
        )}
        <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1
              className="text-3xl sm:text-4xl"
              style={{ fontFamily: "var(--font-serif)" }}
            >
              {player?.player ?? (loadingPlayer ? "Loading…" : "Unknown player")}
            </h1>
            {player && (
              <p className="mt-1 text-muted-foreground">
                {player.season} · {player.squad} · {player.comp}
              </p>
            )}
          </div>
          {player && (
            <div className="flex flex-wrap gap-2">
              <Badge>{player.pos}</Badge>
              <Badge variant="secondary">{Math.round(player.minutes)} minutes</Badge>
              <Badge variant="outline">{player.role}</Badge>
            </div>
          )}
        </div>

        {profile?.archetypes && profile.archetypes.length > 0 && (
          <div className="mb-6 space-y-2">
            <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Archetypes</p>
            {profile.archetypes.map((a) => (
              <div key={a.name} className="flex items-center gap-3">
                <span className="w-40 truncate text-sm">{a.name}</span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full bg-primary"
                    style={{ width: `${Math.round(a.weight * 100)}%` }}
                  />
                </div>
                <span className="w-12 text-right text-xs tabular-nums text-muted-foreground">
                  {Math.round(a.weight * 100)}%
                </span>
              </div>
            ))}
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
          <div className="space-y-6">
            <RadarCard radar={profile?.radar ?? null} loading={loadingPlayer} />
            <SimilarTable
              rows={rows}
              loading={loadingRank}
              error={rankError}
              metric={metric}
            />
          </div>
          <div className="space-y-6">
            <FiltersCard value={filters} onChange={setFilters} />
            <WeightSliders weights={weights} onChange={setWeights} />
          </div>
        </div>
      </main>
    </div>
  );
}
