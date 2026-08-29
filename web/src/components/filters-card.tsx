"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { sliderNumber, sliderPair } from "@/lib/slider-value";

const COMPS = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"] as const;
const POSITIONS = ["DF", "MF", "FW", "GK"] as const;

export type FilterState = {
  eraStart: number;
  eraEnd: number;
  comps: string[];
  positions: string[];
  minMinutes: number;
};

export function FiltersCard({
  value,
  onChange,
}: {
  value: FilterState;
  onChange: (next: FilterState) => void;
}) {
  function toggle(list: string[], item: string) {
    return list.includes(item) ? list.filter((x) => x !== item) : [...list, item];
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Filters</CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        <div>
          <Label className="mb-2 block">
            Era · {value.eraStart - 1}/{String(value.eraStart).slice(2)} – {value.eraEnd - 1}/
            {String(value.eraEnd).slice(2)}
          </Label>
          <Slider
            min={2018}
            max={2025}
            value={[value.eraStart, value.eraEnd]}
            onValueChange={(v) => {
              const [a, b] = sliderPair(v, [value.eraStart, value.eraEnd]);
              onChange({ ...value, eraStart: Math.min(a, b), eraEnd: Math.max(a, b) });
            }}
          />
        </div>
        <div>
          <Label className="mb-2 block">Competition</Label>
          <div className="flex flex-wrap gap-2">
            {COMPS.map((comp) => {
              const on = value.comps.includes(comp);
              return (
                <Button
                  key={comp}
                  size="sm"
                  variant={on || value.comps.length === 0 ? (on ? "default" : "outline") : "outline"}
                  className={value.comps.length === 0 ? "opacity-80" : undefined}
                  onClick={() => onChange({ ...value, comps: toggle(value.comps, comp) })}
                >
                  {comp === "Premier League" ? "Premier League" : comp}
                </Button>
              );
            })}
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            {value.comps.length === 0
              ? "All Big 5 leagues. Click to restrict."
              : "Click again to remove. Empty = all leagues."}
          </p>
        </div>
        <div>
          <Label className="mb-2 block">Position</Label>
          <div className="flex flex-wrap gap-2">
            {POSITIONS.map((pos) => (
              <Badge
                key={pos}
                variant={value.positions.includes(pos) ? "default" : "outline"}
                className="cursor-pointer px-3 py-1"
                onClick={() => onChange({ ...value, positions: toggle(value.positions, pos) })}
              >
                {pos}
              </Badge>
            ))}
          </div>
        </div>
        <div>
          <Label className="mb-2 block">Minutes floor · {value.minMinutes}′</Label>
          <Slider
            min={0}
            max={3000}
            step={90}
            value={[value.minMinutes]}
            onValueChange={(v) => onChange({ ...value, minMinutes: sliderNumber(v, value.minMinutes) })}
          />
        </div>
      </CardContent>
    </Card>
  );
}
