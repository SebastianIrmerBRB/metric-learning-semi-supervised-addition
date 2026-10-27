"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowLeft, ArrowUpRight, Check, ChevronDown, Copy, Plus, Trash2 } from "lucide-react";

import type { TodoFile, TodoItem, TodoPatch } from "@/lib/todo";

// On the static site there is no API; checks and notes are then kept in this browser.
const LOCAL_KEY = "trial-atlas:todo-overrides";

type Overrides = Record<string, { done?: boolean; note?: string }>;

function readOverrides(): Overrides {
  try {
    return JSON.parse(window.localStorage.getItem(LOCAL_KEY) ?? "{}") as Overrides;
  } catch {
    return {};
  }
}

function writeOverrides(value: Overrides) {
  try {
    window.localStorage.setItem(LOCAL_KEY, JSON.stringify(value));
  } catch {
    // Saving locally is a fallback; the list still works for this visit.
  }
}

function applyOverrides(file: TodoFile, overrides: Overrides): TodoFile {
  return {
    ...file,
    items: file.items.map((item) => ({ ...item, ...(overrides[item.id] ?? {}) })),
  };
}

function CommandLine({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="group flex items-start gap-2 rounded-lg border border-white/[0.06] bg-black/40 px-3 py-2">
      <code className="min-w-0 flex-1 whitespace-pre-wrap break-all font-mono text-[11px] leading-relaxed text-[#c8d2e4]">{command}</code>
      <button
        type="button"
        onClick={() => {
          void navigator.clipboard?.writeText(command).then(() => {
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1200);
          });
        }}
        className="focus-ring shrink-0 rounded-md p-1 text-[#6f7c92] hover:bg-white/[0.06] hover:text-white"
        aria-label={`Copy command: ${command}`}
        title="Copy"
      >
        {copied ? <Check size={13} className="text-[#b8e56e]" /> : <Copy size={13} />}
      </button>
    </div>
  );
}

function NoteField({ value, onSave }: { value: string; onSave: (note: string) => void }) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  return (
    <textarea
      value={draft}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={() => {
        if (draft !== value) onSave(draft);
      }}
      rows={draft ? Math.min(6, draft.split("\n").length + 1) : 1}
      placeholder="Note: results, file names, what is left… (saved when you click away)"
      className="focus-ring w-full resize-y rounded-lg border border-white/[0.08] bg-[#0b1222] px-3 py-2 text-xs text-[#e4e9f2] placeholder:text-[#56627a]"
    />
  );
}

function TodoCard({
  item,
  onPatch,
  onDelete,
}: {
  item: TodoItem;
  onPatch: (patch: TodoPatch) => void;
  onDelete: (id: string) => void;
}) {
  const hasDetails = item.steps.length > 0 || item.commands.length > 0;
  const [open, setOpen] = useState(false);
  return (
    <li
      className={`rounded-2xl border p-4 transition ${
        item.done ? "border-white/[0.05] bg-white/[0.01] opacity-60" : "border-white/[0.08] bg-white/[0.025]"
      }`}
    >
      <div className="flex items-start gap-3">
        <button
          type="button"
          role="checkbox"
          aria-checked={item.done}
          aria-label={item.done ? `Mark “${item.title}” as not done` : `Mark “${item.title}” as done`}
          onClick={() => onPatch({ id: item.id, done: !item.done })}
          className={`focus-ring mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-lg border transition ${
            item.done ? "border-[#b8e56e] bg-[#b8e56e] text-[#0b1222]" : "border-white/[0.25] hover:border-[#b8e56e]"
          }`}
        >
          {item.done && <Check size={15} strokeWidth={3} />}
        </button>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-md bg-[#9b8cff]/15 px-1.5 py-0.5 text-[10px] font-bold tabular-nums text-[#c9c1ff]">#{item.rank}</span>
            {item.plan && (
              <span className="rounded-md bg-[#4fd5e7]/10 px-1.5 py-0.5 text-[10px] font-semibold text-[#7fe3f0]">Plan {item.plan}</span>
            )}
            {item.effort && <span className="text-[10px] font-semibold text-[#8895aa]">{item.effort}</span>}
          </div>
          <h3 className={`mt-1.5 text-sm font-semibold leading-snug ${item.done ? "text-[#98a4ba] line-through" : "text-white"}`}>
            {item.title}
          </h3>
          {item.why && <p className="mt-1.5 text-xs leading-relaxed text-[#98a4ba]">{item.why}</p>}

          {(item.links.length > 0 || hasDetails || item.custom) && (
            <div className="mt-3 flex flex-wrap items-center gap-2">
              {item.links.map((link) => (
                <Link
                  key={link.href + link.label}
                  href={link.href}
                  className="focus-ring inline-flex items-center gap-1 rounded-lg border border-[#9b8cff]/30 bg-[#9b8cff]/10 px-2.5 py-1 text-[11px] font-semibold text-[#d8d2ff] hover:bg-[#9b8cff]/20"
                >
                  {link.label}
                  <ArrowUpRight size={12} />
                </Link>
              ))}
              {hasDetails && (
                <button
                  type="button"
                  onClick={() => setOpen((current) => !current)}
                  aria-expanded={open}
                  className="focus-ring inline-flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] font-semibold text-[#8895aa] hover:text-white"
                >
                  <ChevronDown size={13} className={`transition ${open ? "rotate-180" : ""}`} />
                  {open ? "Hide" : "Show"} {item.commands.length ? "steps & commands" : "steps"}
                </button>
              )}
              {item.custom && (
                <button
                  type="button"
                  onClick={() => onDelete(item.id)}
                  className="focus-ring ml-auto inline-flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] font-semibold text-[#8895aa] hover:text-[#f4b4b4]"
                >
                  <Trash2 size={12} />
                  Remove
                </button>
              )}
            </div>
          )}

          {open && hasDetails && (
            <div className="mt-3 space-y-2">
              {item.steps.length > 0 && (
                <ol className="list-decimal space-y-1 pl-5 text-xs leading-relaxed text-[#c8d2e4]">
                  {item.steps.map((step) => (
                    <li key={step}>{step}</li>
                  ))}
                </ol>
              )}
              {item.commands.length > 0 && (
                <div className="space-y-1.5">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#6f7c92]">In ~/metric-learning, with the venv active</p>
                  {item.commands.map((command) => (
                    <CommandLine key={command} command={command} />
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="mt-3">
            <NoteField value={item.note} onSave={(note) => onPatch({ id: item.id, note })} />
          </div>
        </div>
      </div>
    </li>
  );
}

export function TodoBoard({ initial }: { initial: TodoFile | null }) {
  const [file, setFile] = useState<TodoFile | null>(initial);
  // null until the first request says whether the dev server's API is there.
  const [live, setLive] = useState<boolean | null>(null);
  const [hideDone, setHideDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newTitle, setNewTitle] = useState("");
  const [newGroup, setNewGroup] = useState(initial?.groups[0]?.id ?? "");

  useEffect(() => {
    let cancelled = false;
    fetch("/api/todo/", { cache: "no-store" })
      .then((response) => (response.ok ? (response.json() as Promise<TodoFile>) : Promise.reject(new Error(String(response.status)))))
      .then((fresh) => {
        if (cancelled) return;
        setFile(fresh);
        setLive(true);
      })
      .catch(() => {
        if (cancelled) return;
        setLive(false);
        setFile((current) => (current ? applyOverrides(current, readOverrides()) : current));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const send = useCallback(
    async (method: "PATCH" | "POST" | "DELETE", body?: unknown, query = "") => {
      const response = await fetch(`/api/todo/${query}`, {
        method,
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
      const result = (await response.json()) as TodoFile & { error?: string };
      if (!response.ok) throw new Error(result.error ?? `Saving failed (${response.status}).`);
      setFile(result);
      setError(null);
    },
    [],
  );

  const patch = useCallback(
    (change: TodoPatch) => {
      // Show the change at once; the server's answer replaces it.
      setFile((current) =>
        current
          ? {
              ...current,
              items: current.items.map((item) =>
                item.id === change.id
                  ? { ...item, ...(change.done != null ? { done: change.done } : {}), ...(change.note != null ? { note: change.note } : {}) }
                  : item,
              ),
            }
          : current,
      );
      if (live) {
        send("PATCH", change).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
      } else {
        const overrides = readOverrides();
        overrides[change.id] = {
          ...overrides[change.id],
          ...(change.done != null ? { done: change.done } : {}),
          ...(change.note != null ? { note: change.note } : {}),
        };
        writeOverrides(overrides);
      }
    },
    [live, send],
  );

  const remove = useCallback(
    (id: string) => {
      send("DELETE", undefined, `?id=${encodeURIComponent(id)}`).catch((reason: unknown) =>
        setError(reason instanceof Error ? reason.message : String(reason)),
      );
    },
    [send],
  );

  const counts = useMemo(() => {
    const items = file?.items ?? [];
    return { done: items.filter((item) => item.done).length, total: items.length };
  }, [file]);

  if (!file) {
    return (
      <main className="mx-auto max-w-3xl px-5 py-10 text-sm text-[#98a4ba]">
        No todo list yet: <code>react_trial_visualizer/data/todo.json</code> is missing.
      </main>
    );
  }

  const nextUp = [...file.items].filter((item) => !item.done).sort((left, right) => left.rank - right.rank)[0];

  return (
    <main className="mx-auto max-w-3xl px-4 pb-20 pt-8 sm:px-5">
      <Link href="/" className="focus-ring text-xs font-semibold text-[#8895aa] hover:text-white">
        <ArrowLeft size={14} className="mr-1 inline" />
        Back to Trial Atlas
      </Link>

      <header className="mt-5">
        <h1 className="text-2xl font-semibold text-white">{file.title}</h1>
        <p className="mt-2 text-sm leading-relaxed text-[#98a4ba]">{file.subtitle}</p>
        <div className="mt-5 flex items-center gap-3">
          <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/[0.06]" aria-hidden>
            <div
              className="h-full rounded-full bg-[#b8e56e] transition-all"
              style={{ width: `${counts.total ? (counts.done / counts.total) * 100 : 0}%` }}
            />
          </div>
          <span className="text-xs font-semibold tabular-nums text-[#c8d2e4]">
            {counts.done} of {counts.total} done
          </span>
        </div>
        {nextUp && (
          <p className="mt-3 text-xs text-[#c8d2e4]">
            <span className="font-semibold text-[#b8e56e]">Next up:</span> #{nextUp.rank} {nextUp.title}
          </p>
        )}
        <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px] text-[#7f8ba2]">
          <label className="flex cursor-pointer items-center gap-2 font-semibold text-[#c8d2e4]">
            <input type="checkbox" checked={hideDone} onChange={(event) => setHideDone(event.target.checked)} />
            Hide done
          </label>
          <span>
            {live === null
              ? "Connecting…"
              : live
                ? `Saved to react_trial_visualizer/data/todo.json · last change ${file.updatedAt}`
                : "Static site: checks and notes are kept in this browser only"}
          </span>
        </div>
        {error && (
          <p role="alert" className="mt-3 rounded-xl border border-[#d03b3b]/40 bg-[#d03b3b]/10 px-3 py-2 text-xs text-[#f4b4b4]">
            {error}
          </p>
        )}
      </header>

      {file.groups.map((group) => {
        const items = file.items
          .filter((item) => item.group === group.id && !(hideDone && item.done))
          .sort((left, right) => left.rank - right.rank);
        const total = file.items.filter((item) => item.group === group.id);
        if (total.length === 0) return null;
        return (
          <section key={group.id} className="mt-9">
            <div className="flex items-baseline justify-between gap-3">
              <h2 className="text-base font-semibold text-white">{group.title}</h2>
              <span className="text-[11px] tabular-nums text-[#7f8ba2]">
                {total.filter((item) => item.done).length}/{total.length}
              </span>
            </div>
            {group.note && <p className="mt-1 text-xs text-[#7f8ba2]">{group.note}</p>}
            <ul className="mt-3 space-y-3">
              {items.map((item) => (
                <TodoCard key={item.id} item={item} onPatch={patch} onDelete={remove} />
              ))}
              {items.length === 0 && <li className="text-xs text-[#6f7c92]">All done here.</li>}
            </ul>
          </section>
        );
      })}

      {live && (
        <form
          className="mt-10 flex flex-wrap gap-2 rounded-2xl border border-dashed border-white/[0.12] p-3"
          onSubmit={(event) => {
            event.preventDefault();
            if (!newTitle.trim()) return;
            send("POST", { title: newTitle, group: newGroup })
              .then(() => setNewTitle(""))
              .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
          }}
        >
          <input
            value={newTitle}
            onChange={(event) => setNewTitle(event.target.value)}
            placeholder="Add a task of your own…"
            aria-label="New task"
            className="focus-ring min-w-0 flex-1 rounded-lg border border-white/[0.1] bg-[#0b1222] px-3 py-2 text-xs text-white placeholder:text-[#5f6b80]"
          />
          <select
            value={newGroup}
            onChange={(event) => setNewGroup(event.target.value)}
            aria-label="Group"
            className="focus-ring rounded-lg border border-white/[0.1] bg-[#0b1222] px-2 py-2 text-xs text-[#c8d2e4]"
          >
            {file.groups.map((group) => (
              <option key={group.id} value={group.id}>
                {group.title}
              </option>
            ))}
          </select>
          <button
            type="submit"
            disabled={!newTitle.trim()}
            className="focus-ring inline-flex items-center gap-1 rounded-lg border border-[#b8e56e]/35 bg-[#b8e56e]/10 px-3 py-2 text-xs font-semibold text-[#d6f5a4] hover:bg-[#b8e56e]/20 disabled:opacity-40"
          >
            <Plus size={13} />
            Add
          </button>
        </form>
      )}
    </main>
  );
}
