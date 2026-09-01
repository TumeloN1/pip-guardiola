import Link from "next/link";
import { SiteHeader } from "@/components/site-header";

export default function NotFound() {
  return (
    <div className="flex min-h-full flex-1 flex-col">
      <SiteHeader />
      <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-16">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-primary">Not in the index</p>
        <h1 className="mt-3 font-heading text-4xl uppercase tracking-tight text-primary">
          Player-season not found
        </h1>
        <p className="mt-4 max-w-lg text-muted-foreground">
          That player-season is not in the index. It may be below 900 minutes, a cup-only stint, or
          missing advanced FBref stats for that year.
        </p>
        <Link
          href="/"
          className="mt-8 inline-flex h-10 items-center bg-primary px-4 text-sm font-semibold uppercase tracking-wide text-primary-foreground hover:bg-primary/90"
        >
          Back to search
        </Link>
      </main>
    </div>
  );
}
