"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  ArrowLeft,
  Boxes,
  Database,
  Loader2,
  PlayCircle,
  ServerCrash,
  SlidersHorizontal,
} from "lucide-react";
import type { ReactNode } from "react";

import { StudioProvider, useStudio } from "./StudioContext";

const SECTIONS = [
  {
    href: "/studio/spaces",
    label: "Parameter spaces",
    hint: "configs/runs · Optuna search",
    icon: SlidersHorizontal,
  },
  {
    href: "/studio/experiments",
    label: "Experiments",
    hint: "configs/phase_1 · datasets",
    icon: Database,
  },
  {
    href: "/studio/manifests",
    label: "Run manifests",
    hint: "configs/run_manifests",
    icon: Boxes,
  },
  { href: "/studio/launch", label: "Launch", hint: "main.py · launch_runs.py", icon: PlayCircle },
];

function Gate({ children }: { children: ReactNode }) {
  const { loading, error, schema } = useStudio();

  if (loading) {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center gap-3 text-[#8f9bb2]">
        <Loader2 className="size-6 animate-spin text-[var(--violet)]" />
        <p className="text-sm">Importing the training package to read the real schema…</p>
        <p className="text-xs">The first load waits on the Python import; later ones are instant.</p>
      </div>
    );
  }

  if (error || !schema) {
    return (
      <div className="mx-auto max-w-2xl py-16">
        <div className="surface p-6">
          <div className="mb-3 flex items-center gap-2 text-[#ff9a9a]">
            <ServerCrash className="size-5" />
            <h2 className="text-sm font-semibold">The studio backend is not reachable</h2>
          </div>
          <p className="text-xs leading-relaxed text-[#8f9bb2]">
            This page reads and writes files on this machine, so it only works under the local dev
            server. Start it with <code className="font-mono text-[#d9e0ed]">npm run dev</code> in{" "}
            <code className="font-mono text-[#d9e0ed]">react_trial_visualizer</code>, from a shell
            that can reach the project&apos;s Python environment.
          </p>
          {error && (
            <pre className="studio-scroll mt-4 max-h-64 overflow-auto rounded-xl border border-white/[0.08] bg-[#0c1425] p-3 font-mono text-[0.7rem] leading-relaxed text-[#ff9a9a]">
              {error}
            </pre>
          )}
        </div>
      </div>
    );
  }

  return <>{children}</>;
}

export function StudioShell({ children }: { children: ReactNode }) {
  const pathname = usePathname() ?? "";

  return (
    <StudioProvider>
      <div className="mx-auto min-h-screen w-full max-w-[110rem] px-5 py-6 sm:px-8">
        <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <Link
              href="/"
              className="focus-ring mb-2 inline-flex items-center gap-1.5 rounded-lg text-xs text-[#8f9bb2] hover:text-[#d9e0ed]"
            >
              <ArrowLeft className="size-3.5" />
              Trial Atlas
            </Link>
            <h1 className="text-2xl font-semibold tracking-tight text-[#f7f8fc]">Config Studio</h1>
            <p className="mt-1 max-w-2xl text-sm text-[#8f9bb2]">
              Edit search spaces, dataset configs, and run manifests as forms — checked against the
              same validators <code className="font-mono text-xs">main.py</code> uses.
            </p>
          </div>
        </header>

        <nav className="mb-6 flex flex-wrap gap-2">
          {SECTIONS.map((section) => {
            const active = pathname.startsWith(section.href);
            const Icon = section.icon;
            return (
              <Link
                key={section.href}
                href={section.href}
                className={`focus-ring group flex items-center gap-2.5 rounded-2xl border px-3.5 py-2.5 transition-colors ${
                  active
                    ? "border-[var(--violet)]/40 bg-[var(--violet)]/12"
                    : "border-white/[0.08] hover:bg-white/[0.04]"
                }`}
              >
                <Icon
                  className={`size-4 ${active ? "text-[var(--violet)]" : "text-[#8f9bb2] group-hover:text-[#d9e0ed]"}`}
                />
                <span>
                  <span className="block text-xs font-medium text-[#e7ebf4]">{section.label}</span>
                  <span className="block font-mono text-[0.65rem] text-[#8f9bb2]">{section.hint}</span>
                </span>
              </Link>
            );
          })}
        </nav>

        <Gate>{children}</Gate>
      </div>
    </StudioProvider>
  );
}
