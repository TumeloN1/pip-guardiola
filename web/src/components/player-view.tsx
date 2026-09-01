"use client";

import { useEffect, useMemo, useState } from "react";
import { SiteHeader } from "@/components/site-header";
import { PlayerProfileSkeleton } from "@/components/player-profile-skeleton";
import { RadarCard } from "@/components/radar-card";
import { FiltersCard, type FilterState } from "@/components/filters-card";
import { SimilarTable } from "@/components/similar-table";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { getPlayer, getProfile, getSimilar } from "@/lib/api";
import type { HeadlineStat, PlayerHit, ProfileResponse, SimilarResponse, SimilarRow } from "@/lib/types";

const DEFAULT_FILTERS: FilterState = {
  eraStart: 2018,
  eraEnd: 2025,
  comps: ["Premier League"],
  positions: [],
  minMinutes: 900,
};

export function PlayerView({
  id,
  initialPlayer,
  initialProfile,
  initialSimilar,
  loadError: initialError,
}: {
  id: string;
  initialPlayer: PlayerHit | null;
  initialProfile: ProfileResponse | null;
  initialSimilar: SimilarResponse | null;
  loadError: string | null;
}) {
  const [player, setPlayer] = useState<PlayerHit | null>(initialPlayer);
  const [profile, setProfile] = useState<ProfileResponse | null>(initialProfile);
  const [rows, setRows] = useState<SimilarRow[] | null>(initialSimilar?.results ?? null);
  const [metric, setMetric] = useState(initialSimilar?.metric ?? "learned");
  const [loadError, setLoadError] = useState<string | null>(initialError);
  const [rankError, setRankError] = useState<string | null>(null);
  const [loadingPlayer, setLoadingPlayer] = useState(!initialPlayer && !initialError);
  const [loadingRank, setLoadingRank] = useState(!initialSimilar);
  const [filters, setFilters] = useState<FilterState>(() =>
    initialPlayer?.role === "keeper"
      ? { ...DEFAULT_FILTERS, positions: ["GK"], comps: [] }
      : DEFAULT_FILTERS,
  );

  useEffect(() => {
    setPlayer(initialPlayer);
    setProfile(initialProfile);
    setLoadError(initialError);
    setRows(initialSimilar?.results ?? null);
    setMetric(initialSimilar?.metric ?? "learned");
    setLoadingPlayer(!initialPlayer && !initialError);
    setLoadingRank(!initialSimilar);
    if (initialPlayer) {
      setFilters(
        initialPlayer.role === "keeper"
          ? { ...DEFAULT_FILTERS, positions: ["GK"], comps: [] }
          : DEFAULT_FILTERS,
      );
    }
  }, [id, initialPlayer, initialProfile, initialSimilar, initialError]);

  useEffect(() => {
    if (initialPlayer && initialPlayer.id === id) return;
    let cancelled = false;
    setLoadingPlayer(true);
    setLoadError(null);
    Promise.all([getPlayer(id), getProfile(id)])
      .then(([p, prof]) => {
        if (cancelled) return;
        setPlayer(p);
        setProfile(prof);
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
  }, [id, initialPlayer]);

  const similarKey = useMemo(() => JSON.stringify({ id, filters }), [id, filters]);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    setLoadingRank(true);
    setRankError(null);
    const handle = window.setTimeout(() => {
      getSimilar(
        id,
        {
          eraStart: filters.eraStart,
          eraEnd: filters.eraEnd,
          comps: filters.comps,
          positions: filters.positions,
          minMinutes: filters.minMinutes,
          k: 40,
        },
        controller.signal,
      )
        .then((res) => {
          if (cancelled) return;
          setRows(res.results);
          setMetric(res.metric);
        })
        .catch((err: unknown) => {
          if (cancelled || (err instanceof DOMException && err.name === "AbortError")) return;
          setRankError(err instanceof Error ? err.message : "Ranking failed");
          setRows([]);
        })
        .finally(() => {
          if (!cancelled) setLoadingRank(false);
        });
    }, 180);
    return () => {
      cancelled = true;
      controller.abort();
      window.clearTimeout(handle);
    };
  }, [similarKey, id, filters]);

  if (loadingPlayer && !player) {
    return (
      <div className="flex min-h-full flex-1 flex-col">
        <SiteHeader />
        <PlayerProfileSkeleton />
      </div>
    );
  }

  const headline: HeadlineStat[] = profile?.headline ?? [];

  return (
    <div className="flex min-h-full flex-1 flex-col">
      <SiteHeader />
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:py-8">
        {loadError && (
          <div className="mb-6 border border-destructive/40 bg-destructive/10 p-4 text-sm">
            {loadError.includes("unknown")
              ? "That player-season is not in the index. It may be below 900 minutes or missing advanced stats."
              : loadError}
          </div>
        )}
        <div className="mb-6 flex flex-col gap-3 border-b border-border pb-5 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
              Player profile
            </p>
            <h1 className="mt-1 font-heading text-4xl uppercase tracking-tight text-primary sm:text-5xl">
              {player?.player ?? "Unknown player"}
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

        {loadingPlayer && headline.length === 0 ? (
          <div className="mb-8 grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
            {Array.from({ length: 7 }).map((_, i) => (
              <Skeleton key={i} className="h-16 rounded-sm" />
            ))}
          </div>
        ) : headline.length > 0 ? (
          <div className="mb-8 grid grid-cols-2 gap-px overflow-hidden border border-border bg-border sm:grid-cols-4 lg:grid-cols-7">
            {headline.map((stat) => (
              <div key={stat.label} className="bg-card px-3 py-3">
                <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                  {stat.label}
                </p>
                <p className="mt-1 font-heading text-2xl tabular-nums tracking-tight text-primary">
                  {stat.value}
                </p>
              </div>
            ))}
          </div>
        ) : null}

        {profile?.archetypes && profile.archetypes.length > 0 && (
          <div className="mb-8">
            <p className="mb-3 text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
              Playing styles
            </p>
            <div className="grid gap-3 sm:grid-cols-3">
              {profile.archetypes.map((a) => (
                <div key={a.name} className="border border-border bg-card p-4">
                  <div className="flex items-baseline justify-between gap-3">
                    <h2 className="font-heading text-lg uppercase leading-tight tracking-tight text-primary">
                      {a.name}
                    </h2>
                    <span className="shrink-0 text-sm font-semibold tabular-nums text-[#FF2882]">
                      {Math.round(a.weight * 100)}%
                    </span>
                  </div>
                  {a.blurb && <p className="mt-2 text-sm leading-snug text-muted-foreground">{a.blurb}</p>}
                  <div className="mt-3 h-1.5 overflow-hidden bg-muted">
                    <div
                      className="h-full bg-[#00FF85]"
                      style={{ width: `${Math.round(a.weight * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
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
          </div>
        </div>
      </main>
    </div>
  );
}
