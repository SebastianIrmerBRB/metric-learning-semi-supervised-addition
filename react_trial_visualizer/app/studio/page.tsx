import Link from "next/link";
import { Boxes, Database, PlayCircle, SlidersHorizontal } from "lucide-react";

const SECTIONS = [
  {
    href: "/studio/spaces/",
    icon: SlidersHorizontal,
    title: "Parameter spaces",
    body: "Build Optuna search spaces from the real argument surface: every key is checked against a main.py option, a loss or miner constructor argument, or a field of the SSL config it targets.",
    accent: "text-[var(--violet)]",
  },
  {
    href: "/studio/experiments/",
    icon: Database,
    title: "Experiment configs",
    body: "Dataset, split, budget, and the loss/miner arms. Fields carry their argparse help and defaults, and mismatches such as an SSL method that disagrees with its save_dir are flagged.",
    accent: "text-[var(--cyan)]",
  },
  {
    href: "/studio/manifests/",
    icon: Boxes,
    title: "Run manifests",
    body: "Compose the batch launch_runs.py will schedule. Verifying runs the launcher's own loader, so missing configs and reserved options surface before anything starts.",
    accent: "text-[var(--lime)]",
  },
  {
    href: "/studio/launch/",
    icon: PlayCircle,
    title: "Launch",
    body: "Start one main.py process or a whole manifest, preview the exact command first, and follow the log without leaving the page.",
    accent: "text-[#f5cf7a]",
  },
];

export default function StudioPage() {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {SECTIONS.map((section) => {
        const Icon = section.icon;
        return (
          <Link
            key={section.href}
            href={section.href}
            className="focus-ring surface group p-6 transition-colors hover:border-white/20"
          >
            <Icon className={`size-6 ${section.accent}`} />
            <h2 className="mt-4 text-base font-semibold text-[#f7f8fc]">{section.title}</h2>
            <p className="mt-2 text-sm leading-relaxed text-[#8f9bb2]">{section.body}</p>
          </Link>
        );
      })}
    </div>
  );
}
