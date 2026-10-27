/** Page switcher of the hosted build (npm run build:share), in place of the atlas's back link. */
const PAGES = [
  { href: "/", label: "Geometrie" },
  { href: "/embeddings/", label: "Embeddings" },
] as const;

export function ShareNav({ current }: { current: (typeof PAGES)[number]["href"] }) {
  return (
    <nav className="-ml-3 mb-5 flex gap-1 text-xs font-semibold">
      {PAGES.map((page) => (
        <a
          key={page.href}
          href={page.href}
          aria-current={page.href === current ? "page" : undefined}
          className={`focus-ring rounded-lg px-3 py-1.5 ${
            page.href === current ? "bg-white/[0.07] text-white" : "text-[#8895aa] hover:bg-white/[0.05] hover:text-white"
          }`}
        >
          {page.label}
        </a>
      ))}
    </nav>
  );
}
