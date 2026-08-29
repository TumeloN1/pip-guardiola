"use client";

import { PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { RadarPoint } from "@/lib/types";

export function RadarCard({
  radar,
  loading,
}: {
  radar: RadarPoint[] | null;
  loading: boolean;
}) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Percentile profile</CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="mx-auto h-64 w-64 rounded-full" />
        </CardContent>
      </Card>
    );
  }
  if (!radar || radar.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Percentile profile</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          No radar for this player-season.
        </CardContent>
      </Card>
    );
  }
  const data = radar.map((p) => ({
    label: p.label,
    value: p.percentile ?? 0,
  }));
  return (
    <Card>
      <CardHeader>
        <CardTitle>Percentile profile</CardTitle>
        <p className="text-sm text-muted-foreground">
          Versus every outfield (or keeper) season in the index, not just this league-year.
        </p>
      </CardHeader>
      <CardContent className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data} cx="50%" cy="50%" outerRadius="72%">
            <PolarGrid stroke="oklch(0.4 0.03 145)" />
            <PolarAngleAxis
              dataKey="label"
              tick={{ fill: "oklch(0.85 0.03 140)", fontSize: 11 }}
            />
            <Radar
              dataKey="value"
              stroke="oklch(0.84 0.17 132)"
              fill="oklch(0.84 0.17 132)"
              fillOpacity={0.35}
            />
          </RadarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
