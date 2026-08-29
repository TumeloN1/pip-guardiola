"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { searchPlayers } from "@/lib/api";
import type { PlayerHit } from "@/lib/types";

export function PlayerSearch({ autoFocus = false }: { autoFocus?: boolean }) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<PlayerHit[]>([]);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const q = query.trim();
    if (q.length < 2) {
      setHits([]);
      setStatus("idle");
      return;
    }
    const handle = window.setTimeout(async () => {
      setStatus("loading");
      try {
        const rows = await searchPlayers(q);
        setHits(rows);
        setStatus("idle");
        setError(null);
      } catch (err) {
        setStatus("error");
        setError(err instanceof Error ? err.message : "Search failed");
      }
    }, 180);
    return () => window.clearTimeout(handle);
  }, [query]);

  return (
    <Command className="rounded-2xl border border-border bg-card shadow-lg" shouldFilter={false}>
      <CommandInput
        autoFocus={autoFocus}
        placeholder="Search a player — try De Bruyne, Kanté, Haaland…"
        value={query}
        onValueChange={setQuery}
      />
      <CommandList>
        {status === "error" && (
          <div className="px-3 py-4 text-sm text-destructive">
            Couldn’t reach the Kindred API. {error}
          </div>
        )}
        {query.trim().length >= 2 && status === "loading" && hits.length === 0 && (
          <div className="px-3 py-4 text-sm text-muted-foreground">Searching player-seasons…</div>
        )}
        {query.trim().length >= 2 && status !== "loading" && hits.length === 0 && status !== "error" && (
          <CommandEmpty>No player-season matched “{query}”.</CommandEmpty>
        )}
        {hits.length > 0 && (
          <CommandGroup heading="Player-seasons">
            {hits.map((hit) => (
              <CommandItem
                key={hit.id}
                value={hit.id}
                onSelect={() => router.push(`/player/${hit.id}`)}
              >
                <div className="flex min-w-0 flex-1 flex-col">
                  <span className="truncate font-medium">{hit.player}</span>
                  <span className="truncate text-xs text-muted-foreground">
                    {hit.season} · {hit.squad} · {hit.comp}
                  </span>
                </div>
                <span className="text-xs tabular-nums text-muted-foreground">
                  {hit.pos} · {Math.round(hit.minutes)}′
                </span>
              </CommandItem>
            ))}
          </CommandGroup>
        )}
      </CommandList>
    </Command>
  );
}
