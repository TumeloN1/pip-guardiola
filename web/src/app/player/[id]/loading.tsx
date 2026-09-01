import { SiteHeader } from "@/components/site-header";
import { PlayerProfileSkeleton } from "@/components/player-profile-skeleton";

export default function PlayerLoading() {
  return (
    <div className="flex min-h-full flex-1 flex-col">
      <SiteHeader />
      <PlayerProfileSkeleton />
    </div>
  );
}
