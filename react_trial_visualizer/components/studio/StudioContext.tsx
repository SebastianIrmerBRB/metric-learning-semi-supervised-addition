"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { studioApi } from "@/lib/studio/client";
import type { ConfigFileEntry, ConfigKind, StudioSchema } from "@/lib/studio/schema";

interface StudioContextValue {
  schema: StudioSchema | null;
  files: ConfigFileEntry[];
  /** Every config path on disk, so a reference can be checked without a round trip. */
  knownPaths: Set<string>;
  loading: boolean;
  error: string | null;
  refreshFiles: () => Promise<void>;
  filesOfKind: (...kinds: ConfigKind[]) => ConfigFileEntry[];
}

const StudioContext = createContext<StudioContextValue | null>(null);

export function StudioProvider({ children }: { children: ReactNode }) {
  const [schema, setSchema] = useState<StudioSchema | null>(null);
  const [files, setFiles] = useState<ConfigFileEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refreshFiles = useCallback(async () => {
    setFiles(await studioApi.files());
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [loadedSchema, loadedFiles] = await Promise.all([
          studioApi.schema(),
          studioApi.files(),
        ]);
        if (cancelled) return;
        setSchema(loadedSchema);
        setFiles(loadedFiles);
      } catch (caught) {
        if (!cancelled) setError(caught instanceof Error ? caught.message : String(caught));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const value = useMemo<StudioContextValue>(
    () => ({
      schema,
      files,
      knownPaths: new Set(files.map((file) => file.path)),
      loading,
      error,
      refreshFiles,
      filesOfKind: (...kinds: ConfigKind[]) => files.filter((file) => kinds.includes(file.kind)),
    }),
    [schema, files, loading, error, refreshFiles],
  );

  return <StudioContext.Provider value={value}>{children}</StudioContext.Provider>;
}

export function useStudio(): StudioContextValue {
  const value = useContext(StudioContext);
  if (!value) throw new Error("useStudio must be used inside a StudioProvider.");
  return value;
}
