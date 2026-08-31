import Link from "next/link";

export function SiteHeader({ compact = false }: { compact?: boolean }) {
  return (
    <header className="border-b border-border/70 bg-background/80 backdrop-blur">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-4 py-3">
        <Link href="/" className="flex items-baseline gap-2">
          <span
            className="text-xl tracking-tight text-primary sm:text-2xl"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            Pip Guardiola
          </span>
          {!compact && (
            <span className="hidden text-xs uppercase tracking-[0.18em] text-muted-foreground sm:inline">
              playstyle analogues
            </span>
          )}
        </Link>
        <nav className="flex items-center gap-4 text-sm text-muted-foreground">
          <Link href="/map" className="hover:text-foreground">
            Map
          </Link>
          <Link href="/" className="hover:text-foreground">
            Search
          </Link>
        </nav>
      </div>
    </header>
  );
}
