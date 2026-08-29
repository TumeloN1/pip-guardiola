import type {
  MetaResponse,
  PlayerHit,
  ProfileResponse,
  ProjectionPoint,
  SimilarResponse,
} from "./types";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export function searchPlayers(q: string, limit = 12): Promise<PlayerHit[]> {
  const params = new URLSearchParams({ q, limit: String(limit) });
  return getJson(`/api/players?${params}`);
}

export function getMeta(): Promise<MetaResponse> {
  return getJson("/api/meta");
}

export function getPlayer(id: string): Promise<PlayerHit> {
  return getJson(`/api/players/${id}`);
}

export function getProfile(id: string): Promise<ProfileResponse> {
  return getJson(`/api/players/${id}/profile`);
}

export type SimilarParams = {
  eraStart: number;
  eraEnd: number;
  comps: string[];
  positions: string[];
  minMinutes: number;
  k?: number;
  weights: Record<string, number>;
};

export function getSimilar(id: string, params: SimilarParams): Promise<SimilarResponse> {
  const qs = new URLSearchParams({
    era_start: String(params.eraStart),
    era_end: String(params.eraEnd),
    min_minutes: String(params.minMinutes),
    k: String(params.k ?? 15),
  });
  if (params.comps.length) qs.set("comps", params.comps.join(","));
  if (params.positions.length) qs.set("positions", params.positions.join(","));
  if (Object.keys(params.weights).length) {
    qs.set("weights", JSON.stringify(params.weights));
  }
  return getJson(`/api/players/${id}/similar?${qs}`);
}

export function getProjection(role = "outfield"): Promise<{ points: ProjectionPoint[]; source: string }> {
  return getJson(`/api/projection?role=${role}`);
}
