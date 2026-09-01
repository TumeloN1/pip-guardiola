"use client";

import { useEffect, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { ProfileLoadingOverlay } from "@/components/player-profile-skeleton";
import { searchPlayers } from "@/lib/api";
import type { PlayerHit } from "@/lib/types";

export function PlayerSearch({
  autoFocus = false,
  variant = "hero",
}: {
  autoFocus?: boolean;
  variant?: "hero" | "header";
}) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<PlayerHit[]>([]);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const q = query.trim();
    if (q.length < 2) {
      setHits([]);
      setStatus("idle");
      return;
    }
    const controller = new AbortController();
    const handle = window.setTimeout(async () => {
      setStatus("loading");
      try {
        const rows = await searchPlayers(q, 12, controller.signal);
        if (controller.signal.aborted) return;
        setHits(rows);
        setStatus("idle");
        setError(null);
        setOpen(true);
      } catch (err) {
        if (controller.signal.aborted) return;
        setStatus("error");
        setError(err instanceof Error ? err.message : "Search failed");
      }
    }, 160);
    return () => {
      controller.abort();
      window.clearTimeout(handle);
    };
  }, [query]);

  function go(id: string) {
    setQuery("");
    setHits([]);
    setOpen(false);
    startTransition(() => {
      router.push(`/player/${id}`);
    });
  }

  const list = (
    <>
      {status === "error" && (
        <div className="px-3 py-4 text-sm text-destructive">Couldn’t reach the API. {error}</div>
      )}
      {query.trim().length >= 2 && status === "loading" && hits.length === 0 && (
        <div className="px-3 py-4 text-sm text-muted-foreground">Searching player-seasons…</div>
      )}
      {query.trim().length >= 2 && status !== "loading" && hits.length === 0 && status !== "error" && (
        <p className="px-3 py-4 text-sm text-muted-foreground">No player-season matched “{query}”.</p>
      )}
      {hits.length > 0 && (
        <ul className="max-h-72 overflow-y-auto py-1">
          {hits.map((hit) => (
            <li key={hit.id}>
              <button
                type="button"
                className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left hover:bg-muted"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => go(hit.id)}
              >
                <span className="min-w-0 flex-1">
                  <span className="block truncate font-medium text-foreground">{hit.player}</span>
                  <span className="block truncate text-xs text-muted-foreground">
                    {hit.season} · {hit.squad} · {hit.comp}
                  </span>
                </span>
                <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
                  {hit.pos} · {Math.round(hit.minutes)}′
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </>
  );

  if (variant === "header") {
    return (
      <>
        {pending && <ProfileLoadingOverlay />}
        <div className="relative w-full max-w-md">
        <input
          autoFocus={autoFocus}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => hits.length > 0 && setOpen(true)}
          onBlur={() => window.setTimeout(() => setOpen(false), 120)}
          placeholder="Search players"
          className="h-9 w-full border border-white/25 bg-white/10 px-3 text-sm text-white outline-none placeholder:text-white/55 focus:border-[#00FF85] focus:bg-white/15"
        />
        {open && query.trim().length >= 2 && (
          <div className="absolute top-[calc(100%+4px)] z-50 w-full border border-border bg-white text-foreground shadow-lg">
            {list}
          </div>
        )}
        </div>
      </>
    );
  }

  return (
    <>
      {pending && <ProfileLoadingOverlay />}
      <Command className="!rounded-none border border-border bg-card shadow-sm" shouldFilter={false}>
      <CommandInput
        autoFocus={autoFocus}
        placeholder="Search a player — try De Bruyne, Kanté, Haaland…"
        value={query}
        onValueChange={setQuery}
      />
      <CommandList>
        {status === "error" && (
          <div className="px-3 py-4 text-sm text-destructive">Couldn’t reach the API. {error}</div>
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
              <CommandItem key={hit.id} value={hit.id} onSelect={() => go(hit.id)}>
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
    </>
  );
}
