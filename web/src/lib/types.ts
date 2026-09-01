export type PlayerHit = {
  id: string;
  player: string;
  season: string;
  season_end_year: number;
  squad: string;
  comp: string;
  pos: string;
  primary_pos?: string;
  minutes: number;
  role: string;
  fbref_id?: string;
};

export type GroupScore = {
  group: string;
  score: number;
  weight: number;
};

export type SimilarRow = {
  rank: number;
  player_id: string;
  player: string;
  season: string;
  season_end_year: number;
  squad: string;
  comp: string;
  pos: string;
  minutes: number;
  fbref_id?: string;
  similarity: number;
  groups: GroupScore[];
};

export type SimilarResponse = {
  query: PlayerHit;
  results: SimilarRow[];
  metric: string;
};

export type RadarPoint = {
  feature: string;
  percentile: number | null;
  label: string;
};

export type Archetype = {
  name: string;
  weight: number;
  blurb?: string;
};

export type HeadlineStat = {
  label: string;
  value: string;
};

export type ProfileResponse = {
  player: PlayerHit;
  percentiles: Record<string, number | null>;
  radar: RadarPoint[];
  archetypes: Archetype[];
  headline?: HeadlineStat[];
  groups: string[];
};

export type MetaResponse = {
  comps: string[];
  season_end_years: number[];
  positions: string[];
  groups: Record<string, string[]>;
  gk_groups: Record<string, string[]>;
  default_weights: Record<string, number>;
  default_gk_weights: Record<string, number>;
  metric: string;
  examples: { id: string; label: string }[];
};

export type ProjectionPoint = PlayerHit & { x: number; y: number };
