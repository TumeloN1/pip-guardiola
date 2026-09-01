import Link from "next/link";
import { PlayerSearch } from "@/components/player-search";

export function SiteHeader() {
  return (
    <header className="bg-primary text-primary-foreground">
      <div className="h-1 bg-[#00FF85]" />
      <div className="mx-auto flex w-full max-w-6xl items-center gap-4 px-4 py-3">
        <Link href="/" className="shrink-0">
          <span className="font-heading text-2xl uppercase leading-none tracking-tight sm:text-[1.7rem]">
            Pip Guardiola
          </span>
        </Link>
        <div className="hidden min-w-0 flex-1 justify-center sm:flex">
          <PlayerSearch variant="header" />
        </div>
        <nav className="ml-auto flex items-center gap-5 text-xs font-semibold uppercase tracking-[0.16em]">
          <Link href="/" className="hover:text-[#00FF85]">
            Search
          </Link>
          <Link href="/map" className="hover:text-[#00FF85]">
            Map
          </Link>
        </nav>
      </div>
      <div className="border-t border-white/10 px-4 py-2 sm:hidden">
        <PlayerSearch variant="header" />
      </div>
    </header>
  );
}
