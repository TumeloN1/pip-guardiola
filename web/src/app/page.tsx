import Link from "next/link";
import { SiteHeader } from "@/components/site-header";
import { PlayerSearch } from "@/components/player-search";

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
      <section className="bg-primary text-primary-foreground">
        <div className="mx-auto w-full max-w-6xl px-4 py-12 sm:py-16">
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-[#00FF85]">
            Big 5 · 2017-18 to 2024-25
          </p>
          <h1 className="mt-3 max-w-3xl font-heading text-5xl uppercase leading-[0.95] tracking-tight sm:text-7xl">
            Who plays like him?
          </h1>
          <p className="mt-5 max-w-xl text-base text-white/75 sm:text-lg">
            Look up a player-season and rank stylistic analogues — not teammates, not output clones.
            Filter by era, league, position, and minutes.
          </p>
        </div>
      </section>
      <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-10">
        <PlayerSearch autoFocus />
        <div className="mt-6 flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <Link
              key={ex.id}
              href={`/player/${ex.id}`}
              className="border border-border bg-card px-3 py-2 text-sm hover:border-primary hover:text-primary"
            >
              <span className="font-semibold">{ex.label}</span>
              <span className="ml-2 text-muted-foreground">{ex.detail}</span>
            </Link>
          ))}
        </div>
      </main>
    </div>
  );
}
