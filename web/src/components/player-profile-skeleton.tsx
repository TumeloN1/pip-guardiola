import { Skeleton } from "@/components/ui/skeleton";

export function PlayerProfileSkeleton({ message = "Loading profile…" }: { message?: string }) {
  return (
    <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:py-8">
      <div className="mb-6 border-b border-border pb-5">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[#FF2882]">{message}</p>
        <Skeleton className="mt-3 h-12 w-72 rounded-sm sm:w-[28rem]" />
        <Skeleton className="mt-3 h-4 w-56 rounded-sm" />
      </div>
      <div className="mb-8 grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
        {Array.from({ length: 7 }).map((_, i) => (
          <Skeleton key={i} className="h-16 rounded-sm" />
        ))}
      </div>
      <div className="mb-8 grid gap-3 sm:grid-cols-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-28 rounded-sm" />
        ))}
      </div>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="space-y-6">
          <Skeleton className="h-72 rounded-sm" />
          <Skeleton className="h-80 rounded-sm" />
        </div>
        <Skeleton className="h-96 rounded-sm" />
      </div>
    </main>
  );
}

export function ProfileLoadingOverlay() {
  return (
    <div className="fixed inset-0 z-[80] flex flex-col items-center justify-center bg-primary text-primary-foreground">
      <div className="absolute inset-x-0 top-0 h-1 animate-pulse bg-[#00FF85]" />
      <div className="h-12 w-12 animate-spin rounded-full border-4 border-white/25 border-t-[#00FF85]" />
      <p className="mt-5 font-heading text-3xl uppercase tracking-tight">Loading profile</p>
      <p className="mt-2 text-sm text-white/70">Fetching season stats and analogues</p>
    </div>
  );
}
