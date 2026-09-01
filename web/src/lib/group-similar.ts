import type { SimilarRow } from "@/lib/types";

export type SimilarGroup = {
  fbrefId: string;
  player: string;
  best: SimilarRow;
  seasons: SimilarRow[];
};

export function groupSimilarPlayers(rows: SimilarRow[], limit = 10): SimilarGroup[] {
  const order: string[] = [];
  const byId = new Map<string, SimilarGroup>();
  for (const row of rows) {
    const fbrefId = row.fbref_id || row.player_id.split("-")[0];
    const existing = byId.get(fbrefId);
    if (!existing) {
      byId.set(fbrefId, {
        fbrefId,
        player: row.player,
        best: row,
        seasons: [row],
      });
      order.push(fbrefId);
      continue;
    }
    existing.seasons.push(row);
    if (row.similarity > existing.best.similarity) {
      existing.best = row;
    }
  }
  return order.slice(0, limit).map((id) => {
    const group = byId.get(id)!;
    group.seasons = group.seasons
      .slice()
      .sort((a, b) => b.similarity - a.similarity || b.season_end_year - a.season_end_year);
    group.best = group.seasons[0];
    return group;
  });
}
