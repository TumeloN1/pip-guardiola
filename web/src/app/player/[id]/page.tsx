import { PlayerView } from "@/components/player-view";
import type { PlayerHit, ProfileResponse, SimilarResponse } from "@/lib/types";

const API = process.env.KINDRED_API_URL ?? "http://127.0.0.1:8317";

async function readJson<T>(url: string): Promise<T | null> {
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

export default async function PlayerPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const qs = new URLSearchParams({
    era_start: "2018",
    era_end: "2025",
    comps: "Premier League",
    min_minutes: "900",
    k: "15",
  });
  const [player, profile, similar] = await Promise.all([
    readJson<PlayerHit>(`${API}/api/players/${id}`),
    readJson<ProfileResponse>(`${API}/api/players/${id}/profile`),
    readJson<SimilarResponse>(`${API}/api/players/${id}/similar?${qs}`),
  ]);
  return (
    <PlayerView
      id={id}
      initialPlayer={player}
      initialProfile={profile}
      initialSimilar={similar}
      loadError={player ? null : "unknown player-season"}
    />
  );
}
