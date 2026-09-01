import type { ProjectionPoint } from "./types";

export const MAP_COMPS = [
  "Premier League",
  "La Liga",
  "Serie A",
  "Bundesliga",
  "Ligue 1",
] as const;

export const MAP_POSITIONS = ["DF", "MF", "FW"] as const;

export const ERA_MIN = 2018;
export const ERA_MAX = 2025;

export type MapFilterState = {
  comps: string[];
  positions: string[];
  eraStart: number;
  eraEnd: number;
};

export function defaultMapFilters(): MapFilterState {
  return { comps: [], positions: [], eraStart: ERA_MIN, eraEnd: ERA_MAX };
}

export function isDefaultMapFilters(value: MapFilterState): boolean {
  return (
    value.comps.length === 0 &&
    value.positions.length === 0 &&
    value.eraStart === ERA_MIN &&
    value.eraEnd === ERA_MAX
  );
}

export function posTokens(pos: string, primaryPos?: string): string[] {
  return `${pos},${primaryPos ?? ""}`
    .split(",")
    .map((token) => token.trim())
    .filter((token) => token === "DF" || token === "MF" || token === "FW");
}

export function pointMatchesMapFilter(point: ProjectionPoint, filters: MapFilterState): boolean {
  if (filters.comps.length > 0 && !filters.comps.includes(point.comp)) return false;
  if (point.season_end_year < filters.eraStart || point.season_end_year > filters.eraEnd) {
    return false;
  }
  if (filters.positions.length > 0) {
    const tokens = posTokens(point.pos, point.primary_pos);
    if (!filters.positions.some((pos) => tokens.includes(pos))) return false;
  }
  return true;
}

export function seasonLabel(endYear: number): string {
  return `${endYear - 1}/${String(endYear).slice(2)}`;
}
