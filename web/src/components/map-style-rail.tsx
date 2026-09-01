"use client";

import { styleColor } from "@/lib/map-styles";

export function MapStyleRail({
  catalog,
  counts,
  selected,
  visible,
  onToggle,
  onClear,
}: {
  catalog: string[];
  counts: Record<string, number>;
  selected: string[];
  visible: number;
  onToggle: (name: string) => void;
  onClear: () => void;
}) {
  const names = catalog.filter((name) => (counts[name] ?? 0) > 0);
  return (
    <aside className="border border-border bg-card p-4 lg:sticky lg:top-4">
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="font-heading text-lg uppercase tracking-tight text-primary">Archetypes</h2>
        {selected.length > 0 && (
          <button
            type="button"
            className="text-xs font-semibold uppercase tracking-[0.16em] text-primary hover:text-[#FF2882]"
            onClick={onClear}
          >
            Clear
          </button>
        )}
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        Tap a style to isolate that neighbourhood. Multiple allowed.
      </p>
      <div className="mt-3 flex gap-2 overflow-x-auto pb-1 lg:flex-col lg:overflow-visible">
        {names.map((name) => {
          const on = selected.includes(name);
          const color = styleColor(name);
          const n = counts[name] ?? 0;
          const share = visible > 0 ? Math.round((n / visible) * 100) : 0;
          return (
            <button
              key={name}
              type="button"
              onClick={() => onToggle(name)}
              className="shrink-0 border px-3 py-2 text-left text-sm transition-colors lg:w-full"
              style={
                on
                  ? { backgroundColor: color.fill, color: color.ink, borderColor: color.fill }
                  : { borderColor: "var(--border)", backgroundColor: "var(--card)" }
              }
            >
              <span className="flex items-center gap-2">
                {!on && (
                  <span className="size-2 shrink-0" style={{ backgroundColor: color.fill }} />
                )}
                <span className="font-medium leading-tight">{name}</span>
              </span>
              <span className={`mt-0.5 block text-xs tabular-nums ${on ? "opacity-80" : "text-muted-foreground"}`}>
                {n.toLocaleString()} · {share}%
              </span>
            </button>
          );
        })}
      </div>
    </aside>
  );
}
