import Link from "next/link";
import { SiteHeader } from "@/components/site-header";

const CONCEPTS = [
  {
    id: "a",
    title: "A · Territory washes",
    pitch:
      "Four coarse families sit as soft coloured regions behind the dots. Fast to read, but it implies hard borders the metric does not really have. Mobile keeps only two labels.",
    desktop: "/map-concepts/map_concept_a_territories_desktop.png",
    mobile: "/map-concepts/map_concept_a_territories_mobile.png",
  },
  {
    id: "b",
    title: "B · Isolate a neighbourhood",
    pitch:
      "The map stays quiet until you pick a style. A hull and a colour isolate that cluster; several chips can be on at once. Best if you care about one role at a time.",
    desktop: "/map-concepts/map_concept_b_isolate_desktop.png",
    mobile: "/map-concepts/map_concept_b_isolate_mobile.png",
  },
  {
    id: "c",
    title: "C · Atlas labels",
    pitch:
      "No fills. Labels sit on dense neighbourhoods like a tube map, with a note that these are not stat axes. Mobile uses numbered callouts plus a key so type does not collide.",
    desktop: "/map-concepts/map_concept_c_atlas_desktop.png",
    mobile: "/map-concepts/map_concept_c_atlas_mobile.png",
  },
] as const;

export default function MapConceptsPage() {
  return (
    <div className="flex min-h-full flex-1 flex-col">
      <SiteHeader />
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
          <Link href="/map" className="hover:text-primary">
            Style map
          </Link>
          <span className="mx-2">/</span>
          Neighbourhood options
        </p>
        <h1 className="mt-2 font-heading text-4xl uppercase tracking-tight text-primary">
          How should zones read?
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
          League, position, and season filters are already on the live map. These three treatments
          are how we could name the neighbourhoods without putting statistics on the axes. Pick one
          and we will build it against the real projection.
        </p>
        <div className="mt-8 space-y-12">
          {CONCEPTS.map((concept) => (
            <section key={concept.id} className="space-y-4">
              <div>
                <h2 className="font-heading text-2xl uppercase tracking-tight text-primary">
                  {concept.title}
                </h2>
                <p className="mt-1 max-w-2xl text-sm text-muted-foreground">{concept.pitch}</p>
              </div>
              <div className="grid gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(12rem,18rem)]">
                <figure className="border border-border bg-card">
                  <img src={concept.desktop} alt={`${concept.title} desktop`} className="w-full" />
                  <figcaption className="px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                    Desktop
                  </figcaption>
                </figure>
                <figure className="border border-border bg-card">
                  <img
                    src={concept.mobile}
                    alt={`${concept.title} mobile`}
                    className="mx-auto w-full max-w-xs lg:max-w-none"
                  />
                  <figcaption className="px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                    Mobile
                  </figcaption>
                </figure>
              </div>
            </section>
          ))}
        </div>
      </main>
    </div>
  );
}
