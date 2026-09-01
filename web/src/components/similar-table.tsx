"use client";

import { useState } from "react";
import Link from "next/link";
import { ChevronDown } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { groupSimilarPlayers } from "@/lib/group-similar";
import type { SimilarRow } from "@/lib/types";

function topGroups(row: SimilarRow, n = 2): string {
  return row.groups
    .slice()
    .sort((a, b) => b.score - a.score)
    .slice(0, n)
    .map((g) => g.group)
    .join(" · ");
}

export function SimilarTable({
  rows,
  loading,
  error,
  metric,
}: {
  rows: SimilarRow[] | null;
  loading: boolean;
  error: string | null;
  metric: string;
}) {
  const groups = rows ? groupSimilarPlayers(rows, 10) : [];
  const [open, setOpen] = useState<Record<string, boolean>>({});

  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <CardTitle>Closest stylistic matches</CardTitle>
        <p className="text-sm text-muted-foreground">
          Ranked by {metric === "learned" ? "Pip Guardiola’s learned metric" : "z-scored cosine"}.
          Repeat seasons of the same player are grouped — expand a row to see the others.
        </p>
      </CardHeader>
      <CardContent className="px-0">
        {error && (
          <p className="px-6 pb-4 text-sm text-destructive">Couldn’t rank analogues. {error}</p>
        )}
        {loading && (
          <div className="space-y-2 px-6 pb-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} className="h-10 w-full rounded-sm" />
            ))}
          </div>
        )}
        {!loading && rows && rows.length === 0 && !error && (
          <p className="px-6 pb-6 text-sm text-muted-foreground">
            No player-seasons match these filters. Widen the era, drop a league, or lower the
            minutes floor.
          </p>
        )}
        {!loading && groups.length > 0 && (
          <Table>
            <TableHeader className="bg-primary [&_th]:text-primary-foreground">
              <TableRow className="border-primary hover:bg-primary">
                <TableHead className="w-10">#</TableHead>
                <TableHead>Player</TableHead>
                <TableHead className="hidden sm:table-cell">Best season</TableHead>
                <TableHead className="hidden md:table-cell">Club</TableHead>
                <TableHead>Sim</TableHead>
                <TableHead className="hidden lg:table-cell">Why</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {groups.map((group, index) => {
                const extra = group.seasons.length - 1;
                const expanded = Boolean(open[group.fbrefId]);
                return (
                  <SimilarGroupRows
                    key={group.fbrefId}
                    rank={index + 1}
                    group={group}
                    extra={extra}
                    expanded={expanded}
                    onToggle={() =>
                      setOpen((prev) => ({ ...prev, [group.fbrefId]: !prev[group.fbrefId] }))
                    }
                  />
                );
              })}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}

function SimilarGroupRows({
  rank,
  group,
  extra,
  expanded,
  onToggle,
}: {
  rank: number;
  group: ReturnType<typeof groupSimilarPlayers>[number];
  extra: number;
  expanded: boolean;
  onToggle: () => void;
}) {
  const row = group.best;
  return (
    <>
      <TableRow>
        <TableCell className="tabular-nums text-muted-foreground">{rank}</TableCell>
        <TableCell>
          <Link href={`/player/${row.player_id}`} className="font-semibold text-primary hover:text-[#FF2882]">
            {row.player}
          </Link>
          <div className="text-xs text-muted-foreground sm:hidden">
            {row.season} · {row.squad}
          </div>
          {extra > 0 && (
            <button
              type="button"
              onClick={onToggle}
              className="mt-1 inline-flex items-center gap-1 text-xs font-medium text-[#FF2882] hover:underline"
              aria-expanded={expanded}
            >
              <ChevronDown className={`size-3.5 transition-transform ${expanded ? "rotate-180" : ""}`} />
              {expanded ? "Hide" : `+${extra} other ${extra === 1 ? "season" : "seasons"}`}
            </button>
          )}
        </TableCell>
        <TableCell className="hidden sm:table-cell">{row.season}</TableCell>
        <TableCell className="hidden md:table-cell">
          {row.squad}
          <span className="block text-xs text-muted-foreground">{row.comp}</span>
        </TableCell>
        <TableCell className="tabular-nums">{row.similarity.toFixed(3)}</TableCell>
        <TableCell className="hidden lg:table-cell">
          <Badge variant="secondary" className="font-normal">
            {topGroups(row)}
          </Badge>
        </TableCell>
      </TableRow>
      {expanded &&
        group.seasons.slice(1).map((season) => (
          <TableRow key={season.player_id} className="bg-muted/40">
            <TableCell />
            <TableCell>
              <Link
                href={`/player/${season.player_id}`}
                className="text-sm text-primary hover:text-[#FF2882]"
              >
                {season.season}
              </Link>
              <div className="text-xs text-muted-foreground sm:hidden">{season.squad}</div>
            </TableCell>
            <TableCell className="hidden sm:table-cell text-sm">{season.season}</TableCell>
            <TableCell className="hidden md:table-cell text-sm">
              {season.squad}
              <span className="block text-xs text-muted-foreground">{season.comp}</span>
            </TableCell>
            <TableCell className="tabular-nums text-sm">{season.similarity.toFixed(3)}</TableCell>
            <TableCell className="hidden lg:table-cell">
              <Badge variant="outline" className="font-normal">
                {topGroups(season)}
              </Badge>
            </TableCell>
          </TableRow>
        ))}
    </>
  );
}
