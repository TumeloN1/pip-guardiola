"use client";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  ERA_MAX,
  ERA_MIN,
  MAP_COMPS,
  MAP_POSITIONS,
  isDefaultMapFilters,
  seasonLabel,
  type MapFilterState,
} from "@/lib/map-filters";
import { sliderPair } from "@/lib/slider-value";

export function MapFilters({
  value,
  onChange,
  visible,
  total,
}: {
  value: MapFilterState;
  onChange: (next: MapFilterState) => void;
  visible: number;
  total: number;
}) {
  function toggle(list: string[], item: string) {
    return list.includes(item) ? list.filter((x) => x !== item) : [...list, item];
  }

  return (
    <div className="mt-4 space-y-4 border border-border bg-card p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
          {visible.toLocaleString()} of {total.toLocaleString()} seasons
        </p>
        {!isDefaultMapFilters(value) && (
          <button
            type="button"
            className="text-xs font-semibold uppercase tracking-[0.16em] text-primary hover:text-[#FF2882]"
            onClick={() => onChange({ comps: [], positions: [], eraStart: ERA_MIN, eraEnd: ERA_MAX })}
          >
            Reset
          </button>
        )}
      </div>
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.4fr)_auto_minmax(0,1fr)]">
        <div>
          <Label className="mb-2 block text-xs font-semibold uppercase tracking-[0.16em]">
            League
          </Label>
          <div className="flex flex-wrap gap-2">
            {MAP_COMPS.map((comp) => {
              const on = value.comps.includes(comp);
              const all = value.comps.length === 0;
              return (
                <Button
                  key={comp}
                  size="sm"
                  variant={on || all ? (on ? "default" : "outline") : "outline"}
                  className={all ? "opacity-80" : undefined}
                  onClick={() => onChange({ ...value, comps: toggle(value.comps, comp) })}
                >
                  {comp}
                </Button>
              );
            })}
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            {value.comps.length === 0 ? "All Big 5. Click to restrict." : "Empty selection shows every league."}
          </p>
        </div>
        <div>
          <Label className="mb-2 block text-xs font-semibold uppercase tracking-[0.16em]">
            Position
          </Label>
          <div className="flex flex-wrap gap-2">
            {MAP_POSITIONS.map((pos) => {
              const on = value.positions.includes(pos);
              return (
                <Button
                  key={pos}
                  size="sm"
                  variant={on || value.positions.length === 0 ? (on ? "default" : "outline") : "outline"}
                  className={value.positions.length === 0 ? "opacity-80" : undefined}
                  onClick={() => onChange({ ...value, positions: toggle(value.positions, pos) })}
                >
                  {pos}
                </Button>
              );
            })}
          </div>
          <p className="mt-2 text-xs text-muted-foreground">Hybrids such as DF,MF match either chip.</p>
        </div>
        <div>
          <Label className="mb-2 block text-xs font-semibold uppercase tracking-[0.16em]">
            Season · {seasonLabel(value.eraStart)} – {seasonLabel(value.eraEnd)}
          </Label>
          <Slider
            min={ERA_MIN}
            max={ERA_MAX}
            value={[value.eraStart, value.eraEnd]}
            onValueChange={(next) => {
              const [a, b] = sliderPair(next, [value.eraStart, value.eraEnd]);
              onChange({ ...value, eraStart: Math.min(a, b), eraEnd: Math.max(a, b) });
            }}
          />
        </div>
      </div>
    </div>
  );
}
