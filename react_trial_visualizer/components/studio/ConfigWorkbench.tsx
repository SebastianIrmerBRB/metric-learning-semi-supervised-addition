"use client";

import {
  Copy,
  FileJson,
  FilePlus2,
  Loader2,
  RefreshCw,
  Save,
  Search,
  ShieldCheck,
  Trash2,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";

import { studioApi, type ValidationResult } from "@/lib/studio/client";
import { countIssues } from "@/lib/studio/lint";
import type { ConfigFileEntry, ConfigKind, Issue } from "@/lib/studio/schema";

import { useStudio } from "./StudioContext";
import { Button, IssueBadge, IssueList } from "./ui";

type Document = Record<string, unknown>;

interface ConfigWorkbenchProps {
  /** Only files holding this kind of document are listed by default. */
  kind: ConfigKind;
  /** Directory a new file of this kind is suggested in. */
  defaultDirectory: string;
  emptyDocument: () => Document;
  validateKind: "hparam" | "experiment" | "manifest";
  /** Extra fields the Python validator needs, such as the directory `extends` resolves against. */
  validatePayload?: (document: Document, path: string | null) => Record<string, unknown>;
  lint: (document: Document) => Issue[];
  children: (context: {
    document: Document;
    setDocument: (next: Document) => void;
    issues: Issue[];
    path: string | null;
    serverResult: ValidationResult | null;
  }) => ReactNode;
  sidebarFooter?: ReactNode;
}

function groupByDirectory(files: ConfigFileEntry[]): [string, ConfigFileEntry[]][] {
  const groups = new Map<string, ConfigFileEntry[]>();
  for (const file of files) {
    const bucket = groups.get(file.directory) ?? [];
    bucket.push(file);
    groups.set(file.directory, bucket);
  }
  return [...groups.entries()].sort((left, right) => left[0].localeCompare(right[0]));
}

export function ConfigWorkbench({
  kind,
  defaultDirectory,
  emptyDocument,
  validateKind,
  validatePayload,
  lint,
  children,
  sidebarFooter,
}: ConfigWorkbenchProps) {
  const { files: allFiles, refreshFiles } = useStudio();

  const [path, setPath] = useState<string | null>(null);
  const [document, setDocumentState] = useState<Document>(() => emptyDocument());
  const [savedSnapshot, setSavedSnapshot] = useState<string>("");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [busy, setBusy] = useState<"load" | "save" | "verify" | "delete" | null>(null);
  const [serverResult, setServerResult] = useState<ValidationResult | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [showRaw, setShowRaw] = useState(false);
  const [rawDraft, setRawDraft] = useState<string | null>(null);
  const [showAllKinds, setShowAllKinds] = useState(false);

  const files = useMemo(
    () => allFiles.filter((file) => showAllKinds || file.kind === kind),
    [allFiles, kind, showAllKinds],
  );

  const otherKindCount = allFiles.length - allFiles.filter((file) => file.kind === kind).length;

  const visibleFiles = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return files;
    return files.filter((file) => file.path.toLowerCase().includes(needle));
  }, [files, query]);

  const openedKind = allFiles.find((file) => file.path === path)?.kind;
  const kindMismatch = path !== null && openedKind !== undefined && openedKind !== kind;

  const dirty = JSON.stringify(document) !== savedSnapshot;
  const issues = useMemo(() => lint(document), [document, lint]);
  const totals = countIssues(issues);

  const setDocument = useCallback((next: Document) => {
    setDocumentState(next);
    setServerResult(null);
    setNotice(null);
  }, []);

  const openFile = useCallback(
    async (nextPath: string) => {
      setBusy("load");
      setLoadError(null);
      setNotice(null);
      setServerResult(null);
      try {
        const file = await studioApi.readFile(nextPath);
        if (file.parseError) {
          setLoadError(`${nextPath} is not valid JSON: ${file.parseError}`);
          return;
        }
        const loaded = (file.data ?? {}) as Document;
        setPath(nextPath);
        setDocumentState(loaded);
        setSavedSnapshot(JSON.stringify(loaded));
        setRawDraft(null);
      } catch (error) {
        setLoadError(error instanceof Error ? error.message : String(error));
      } finally {
        setBusy(null);
      }
    },
    [],
  );

  useEffect(() => {
    if (path === null && files.length > 0 && savedSnapshot === "") {
      void openFile(files[0].path);
    }
  }, [path, files, savedSnapshot, openFile]);

  const verify = useCallback(async () => {
    setBusy("verify");
    setNotice(null);
    try {
      const payload = validatePayload?.(document, path) ?? { data: document, path };
      const result = await studioApi.validate<ValidationResult>(validateKind, payload);
      setServerResult(result);
    } catch (error) {
      setServerResult({ valid: false, errors: [error instanceof Error ? error.message : String(error)] });
    } finally {
      setBusy(null);
    }
  }, [document, path, validateKind, validatePayload]);

  const save = useCallback(
    async (targetPath: string) => {
      setBusy("save");
      setNotice(null);
      try {
        const response = await studioApi.writeFile(targetPath, document);
        setPath(targetPath);
        setSavedSnapshot(JSON.stringify(document));
        await refreshFiles();
        setNotice(
          response.backup
            ? `Saved. The previous version is in ${response.backup}.`
            : "Saved as a new file.",
        );
      } catch (error) {
        setLoadError(error instanceof Error ? error.message : String(error));
      } finally {
        setBusy(null);
      }
    },
    [document, refreshFiles],
  );

  const createNew = useCallback(() => {
    const suggestion = `${defaultDirectory}/new-config.json`;
    const requested = window.prompt("New config path (relative to the repo root)", suggestion);
    if (!requested) return;
    const fresh = emptyDocument();
    setPath(requested.trim());
    setDocumentState(fresh);
    setSavedSnapshot("");
    setServerResult(null);
    setNotice("Unsaved. Adjust the fields, then press Save.");
  }, [emptyDocument, defaultDirectory]);

  const duplicate = useCallback(() => {
    if (!path) return;
    const suggestion = path.replace(/\.json$/, "-copy.json");
    const requested = window.prompt("Save a copy as", suggestion);
    if (!requested) return;
    void save(requested.trim());
  }, [path, save]);

  const remove = useCallback(async () => {
    if (!path) return;
    if (!window.confirm(`Delete ${path}? A timestamped copy is kept in configs/.studio-backups.`)) return;
    setBusy("delete");
    try {
      const response = await studioApi.deleteFile(path);
      await refreshFiles();
      setPath(null);
      setSavedSnapshot("");
      setDocumentState(emptyDocument());
      setNotice(`Deleted. A copy is in ${response.backup}.`);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(null);
    }
  }, [path, refreshFiles, emptyDocument]);

  return (
    <div className="grid gap-5 lg:grid-cols-[19rem_minmax(0,1fr)]">
      <aside className="surface flex max-h-[calc(100vh-14rem)] flex-col p-4 lg:sticky lg:top-6">
        <div className="mb-3 flex items-center gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-[#8f9bb2]" />
            <input
              className="studio-input pl-8 text-xs"
              placeholder="Filter configs…"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </div>
          <button
            type="button"
            title="New config"
            onClick={createNew}
            className="focus-ring flex size-[2.35rem] shrink-0 items-center justify-center rounded-[0.7rem] border border-white/[0.12] text-[#d9e0ed] hover:bg-white/[0.05]"
          >
            <FilePlus2 className="size-4" />
          </button>
        </div>

        {otherKindCount > 0 && (
          <button
            type="button"
            onClick={() => setShowAllKinds((value) => !value)}
            className="focus-ring mb-2 rounded-lg px-1 text-left text-[0.7rem] text-[#8f9bb2] hover:text-[#d9e0ed]"
          >
            {showAllKinds
              ? `Showing all ${allFiles.length} configs · show only ${kind}`
              : `${otherKindCount} other configs hidden · show all`}
          </button>
        )}

        <div className="studio-scroll -mr-2 flex-1 overflow-y-auto pr-2">
          {groupByDirectory(visibleFiles).map(([directory, entries]) => (
            <div key={directory} className="mb-3">
              <p className="mb-1 truncate font-mono text-[0.65rem] uppercase tracking-wide text-[#8f9bb2]">
                {directory}
              </p>
              <ul className="space-y-0.5">
                {entries.map((file) => (
                  <li key={file.path}>
                    <button
                      type="button"
                      onClick={() => openFile(file.path)}
                      className={`focus-ring flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-xs transition-colors ${
                        file.path === path
                          ? "bg-[var(--violet)]/15 text-[#f7f8fc]"
                          : "text-[#b9c2d4] hover:bg-white/[0.04]"
                      }`}
                    >
                      <FileJson className="size-3.5 shrink-0 opacity-60" />
                      <span className="truncate font-mono">{file.name}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
          {visibleFiles.length === 0 && (
            <p className="px-2 py-6 text-center text-xs text-[#8f9bb2]">No configs match.</p>
          )}
        </div>

        {sidebarFooter && <div className="mt-3 border-t border-white/[0.06] pt-3">{sidebarFooter}</div>}
      </aside>

      <div className="min-w-0 space-y-5">
        <div className="surface flex flex-wrap items-center gap-3 px-5 py-3.5">
          <div className="min-w-0 flex-1">
            <p className="truncate font-mono text-xs text-[#e7ebf4]">{path ?? "No file selected"}</p>
            <p className="mt-0.5 text-[0.7rem] text-[#8f9bb2]">
              {dirty ? "Unsaved changes" : "In sync with disk"}
              {totals.error > 0 && ` · ${totals.error} error${totals.error === 1 ? "" : "s"}`}
              {totals.warning > 0 && ` · ${totals.warning} warning${totals.warning === 1 ? "" : "s"}`}
            </p>
          </div>
          <IssueBadge issues={issues} />
          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={() => setShowRaw((value) => !value)} title="Toggle the raw JSON view">
              <FileJson className="size-3.5" />
              {showRaw ? "Form" : "JSON"}
            </Button>
            <Button onClick={verify} disabled={busy !== null}>
              {busy === "verify" ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <ShieldCheck className="size-3.5" />
              )}
              Verify
            </Button>
            <Button onClick={duplicate} disabled={!path}>
              <Copy className="size-3.5" />
              Copy
            </Button>
            <Button onClick={remove} variant="danger" disabled={!path || busy !== null}>
              <Trash2 className="size-3.5" />
            </Button>
            <Button
              variant="primary"
              disabled={!path || busy !== null || !dirty}
              onClick={() => path && save(path)}
            >
              {busy === "save" ? <Loader2 className="size-3.5 animate-spin" /> : <Save className="size-3.5" />}
              Save
            </Button>
          </div>
        </div>

        {kindMismatch && (
          <div className="surface-soft border-[#f5cf7a]/30 px-4 py-3 text-xs text-[#f5cf7a]">
            This file reads as a <strong>{openedKind}</strong> config, not a {kind} config. The form
            below will not match it — open it in the matching section instead.
          </div>
        )}

        {(loadError || notice) && (
          <div
            className={`surface-soft px-4 py-3 text-xs ${loadError ? "text-[#ff9a9a]" : "text-[#b8e56e]"}`}
          >
            {loadError ?? notice}
          </div>
        )}

        {serverResult && (
          <div className="surface p-5">
            <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-[#f7f8fc]">
              <ShieldCheck className="size-4 text-[var(--cyan)]" />
              Checked with the project&apos;s own loader
            </h3>
            {serverResult.valid ? (
              <p className="text-xs text-[#b8e56e]">
                Accepted. This document loads exactly as written.
              </p>
            ) : (
              <ul className="space-y-1.5">
                {serverResult.errors.map((message, index) => (
                  <li key={index} className="font-mono text-[0.7rem] leading-relaxed text-[#ff9a9a]">
                    {message}
                  </li>
                ))}
              </ul>
            )}
            {serverResult.commands && serverResult.commands.length > 0 && (
              <div className="mt-4 space-y-2">
                <p className="studio-label">Resolved commands</p>
                {serverResult.commands.map((entry) => (
                  <div key={entry.name} className="surface-soft px-3 py-2">
                    <p className="font-mono text-[0.7rem] text-[var(--cyan)]">{entry.name}</p>
                    <p className="studio-scroll mt-1 overflow-x-auto whitespace-pre font-mono text-[0.7rem] text-[#b9c2d4]">
                      {entry.command.join(" ")}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {showRaw ? (
          <div className="surface p-5">
            <div className="mb-3 flex items-center justify-between gap-3">
              <h3 className="text-sm font-semibold text-[#f7f8fc]">Raw JSON</h3>
              <Button
                onClick={() => {
                  if (rawDraft === null) return;
                  try {
                    setDocument(JSON.parse(rawDraft) as Document);
                    setRawDraft(null);
                    setLoadError(null);
                  } catch (error) {
                    setLoadError(error instanceof Error ? error.message : String(error));
                  }
                }}
                disabled={rawDraft === null}
              >
                <RefreshCw className="size-3.5" />
                Apply to form
              </Button>
            </div>
            <textarea
              className="studio-input min-h-[26rem] text-xs"
              spellCheck={false}
              value={rawDraft ?? JSON.stringify(document, null, 2)}
              onChange={(event) => setRawDraft(event.target.value)}
            />
          </div>
        ) : (
          children({ document, setDocument, issues, path, serverResult })
        )}

        <div className="surface p-5">
          <h3 className="mb-3 text-sm font-semibold text-[#f7f8fc]">Checks</h3>
          <IssueList issues={issues} empty="Nothing looks off in this config." />
        </div>
      </div>
    </div>
  );
}
