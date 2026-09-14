import type { Metadata } from "next";
import type { ReactNode } from "react";

import { StudioShell } from "@/components/studio/StudioShell";

export const metadata: Metadata = {
  title: "Config Studio · Trial Atlas",
  description: "Edit search spaces, experiment configs, and run manifests with validation.",
};

export default function StudioLayout({ children }: { children: ReactNode }) {
  return <StudioShell>{children}</StudioShell>;
}
