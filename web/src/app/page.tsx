import Link from "next/link";
import { SiteHeader } from "@/components/site-header";
import { PlayerSearch } from "@/components/player-search";
import { Badge } from "@/components/ui/badge";

const EXAMPLES = [
  { id: "e46012d4-2020", label: "Kevin De Bruyne", detail: "2019-20 · Manchester City" },
  { id: "e342ad68-2020", label: "Mohamed Salah", detail: "2019-20 · Liverpool" },
  { id: "e06683ca-2019", label: "Virgil van Dijk", detail: "2018-19 · Liverpool" },
  { id: "1f44ac21-2023", label: "Erling Haaland", detail: "2022-23 · Manchester City" },
];

export default function HomePage() {
  return (
    <div className="flex min-h-full flex-1 flex-col">
      <SiteHeader />
      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-center px-4 py-12 sm:py-20">
        <p className="mb-3 text-xs uppercase tracking-[0.22em] text-primary">Big 5 · 2017-18 to 2024-25</p>
        <h1
          className="text-4xl leading-tight text-foreground sm:text-6xl"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          Who plays like him?
        </h1>
        <p className="mt-4 max-w-xl text-base text-muted-foreground sm:text-lg">
          Pip Guardiola looks up a player-season and ranks stylistic analogues — not teammates, not
          output clones. Filter by era, league, position, and minutes. Re-weight finishing,
          carrying, or defending live.
        </p>
        <div className="mt-8">
          <PlayerSearch autoFocus />
        </div>
        <div className="mt-8 flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <Link key={ex.id} href={`/player/${ex.id}`}>
              <Badge variant="secondary" className="cursor-pointer px-3 py-1.5 text-sm font-normal">
                <span className="font-medium">{ex.label}</span>
                <span className="ml-2 text-muted-foreground">{ex.detail}</span>
              </Badge>
            </Link>
          ))}
        </div>
      </main>
    </div>
  );
}
