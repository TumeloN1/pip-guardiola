import { notFound } from "next/navigation";
import { PlayerView } from "@/components/player-view";
import type { PlayerHit, ProfileResponse } from "@/lib/types";

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
  const [player, profile] = await Promise.all([
    readJson<PlayerHit>(`${API}/api/players/${id}`),
    readJson<ProfileResponse>(`${API}/api/players/${id}/profile`),
  ]);
  if (!player) notFound();
  return (
    <PlayerView
      id={id}
      initialPlayer={player}
      initialProfile={profile}
      initialSimilar={null}
      loadError={null}
    />
  );
}
