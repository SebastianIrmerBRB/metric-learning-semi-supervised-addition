"use client";

import type { ConfigFileEntry, LaunchStatus, StudioSchema } from "./schema";

export interface ApiFailure {
  ok: false;
  error: string;
}

async function request<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await fetch(input, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  const payload = (await response.json()) as T & { ok?: boolean; error?: string };
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error ?? `Request failed: ${response.status}`);
  }
  return payload;
}

export const studioApi = {
  schema: () => request<{ schema: StudioSchema }>("/api/studio/schema/").then((body) => body.schema),

  files: () =>
    request<{ files: ConfigFileEntry[] }>("/api/studio/files/").then((body) => body.files),

  readFile: (path: string) =>
    request<{ path: string; raw: string; data: unknown; parseError: string | null; modified: string }>(
      `/api/studio/file/?path=${encodeURIComponent(path)}`,
    ),

  writeFile: (path: string, data: unknown) =>
    request<{ path: string; backup: string | null; modified: string }>("/api/studio/file/", {
      method: "PUT",
      body: JSON.stringify({ path, data }),
    }),

  deleteFile: (path: string) =>
    request<{ path: string; backup: string | null }>(
      `/api/studio/file/?path=${encodeURIComponent(path)}`,
      { method: "DELETE" },
    ),

  validate: <T,>(kind: "hparam" | "experiment" | "manifest" | "inspect", payload: Record<string, unknown>) =>
    request<{ result: T }>("/api/studio/validate/", {
      method: "POST",
      body: JSON.stringify({ kind, ...payload }),
    }).then((body) => body.result),

  launches: () => request<{ launches: LaunchStatus[] }>("/api/studio/launch/").then((body) => body.launches),

  launch: (payload: { kind: "main" | "scheduler"; label: string; args: string[]; dryRun?: boolean }) =>
    request<{ launch: LaunchStatus }>("/api/studio/launch/", {
      method: "POST",
      body: JSON.stringify(payload),
    }).then((body) => body.launch),

  stopLaunch: (id: string, signal?: "SIGTERM" | "SIGKILL") =>
    request<{ stopped: boolean }>("/api/studio/launch/stop/", {
      method: "POST",
      body: JSON.stringify({ id, signal }),
    }),

  clearFinished: () => request<{ removed: number }>("/api/studio/launch/", { method: "DELETE" }),

  launchLog: (id: string) =>
    request<{ text: string; size: number }>(`/api/studio/launch/log/?id=${encodeURIComponent(id)}`),
};

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  resolved?: Record<string, unknown>;
  commands?: { name: string; command: string[] }[];
}
