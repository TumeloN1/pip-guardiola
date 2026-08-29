"use client";

import Link from "next/link";
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
  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <CardTitle>Closest stylistic matches</CardTitle>
        <p className="text-sm text-muted-foreground">
          Ranked by {metric === "learned" ? "the learned Kindred metric" : "z-scored cosine"} ·
          columns explain which feature groups lined up
        </p>
      </CardHeader>
      <CardContent className="px-0">
        {error && (
          <p className="px-6 pb-4 text-sm text-destructive">Couldn’t rank analogues. {error}</p>
        )}
        {loading && (
          <div className="space-y-2 px-6 pb-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} className="h-10 w-full" />
            ))}
          </div>
        )}
        {!loading && rows && rows.length === 0 && !error && (
          <p className="px-6 pb-6 text-sm text-muted-foreground">
            No player-seasons match these filters. Widen the era, drop a league, or lower the
            minutes floor.
          </p>
        )}
        {!loading && rows && rows.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-10">#</TableHead>
                <TableHead>Player</TableHead>
                <TableHead className="hidden sm:table-cell">Season</TableHead>
                <TableHead className="hidden md:table-cell">Club</TableHead>
                <TableHead>Sim</TableHead>
                <TableHead className="hidden lg:table-cell">Why</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => (
                <TableRow key={`${row.player_id}-${row.rank}`}>
                  <TableCell className="tabular-nums text-muted-foreground">{row.rank}</TableCell>
                  <TableCell>
                    <Link href={`/player/${row.player_id}`} className="font-medium hover:text-primary">
                      {row.player}
                    </Link>
                    <div className="text-xs text-muted-foreground sm:hidden">
                      {row.season} · {row.squad}
                    </div>
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
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
