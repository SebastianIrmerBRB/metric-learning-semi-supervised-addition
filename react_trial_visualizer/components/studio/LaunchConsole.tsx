"use client";

import {
  Circle,
  Eraser,
  Loader2,
  PlayCircle,
  RefreshCw,
  ScrollText,
  Square,
  Terminal,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { studioApi } from "@/lib/studio/client";
import type { LaunchStatus } from "@/lib/studio/schema";

import { ArgsBuilder } from "./ArgsBuilder";
import { useStudio } from "./StudioContext";
import { Button, Card, Field, NumberInput, Select, TextInput, Toggle } from "./ui";

function CommandPreview({ command }: { command: string[] }) {
  return (
    <div className="surface-soft studio-scroll overflow-x-auto px-3 py-2.5">
      <p className="whitespace-pre font-mono text-[0.7rem] leading-relaxed text-[#b9c2d4]">
        <span className="text-[var(--lime)]">$</span> {command.join(" ")}
      </p>
    </div>
  );
}

function SingleRunPanel({ onLaunched }: { onLaunched: () => void }) {
  const { schema, filesOfKind } = useStudio();
  const [experimentConfig, setExperimentConfig] = useState<string | null>(null);
  const [hparamConfig, setHparamConfig] = useState<string | null>(null);
  const [saveDir, setSaveDir] = useState("");
  const [device, setDevice] = useState<string | null>("cuda");
  const [sslDevice, setSslDevice] = useState<string | null>(null);
  const [extraArgs, setExtraArgs] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const args = useMemo(() => {
    const tokens: string[] = [];
    if (experimentConfig) tokens.push("--experiment-config", experimentConfig);
    if (hparamConfig) tokens.push("--hparam_config", hparamConfig);
    if (saveDir.trim()) tokens.push("--save_dir", saveDir.trim());
    if (device) tokens.push("--device", device);
    if (sslDevice) tokens.push("--ssl-device", sslDevice);
    return [...tokens, ...extraArgs];
  }, [experimentConfig, hparamConfig, saveDir, device, sslDevice, extraArgs]);

  const launch = async () => {
    setBusy(true);
    setError(null);
    try {
      await studioApi.launch({
        kind: "main",
        label: saveDir.trim() || experimentConfig?.split("/").pop() || "main.py",
        args,
      });
      onLaunched();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  if (!schema) return null;

  return (
    <Card
      title="One training process"
      subtitle="Runs main.py directly. The experiment config supplies the dataset and budget; the search config supplies the study."
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Experiment config">
          <Select
            value={experimentConfig}
            onChange={setExperimentConfig}
            options={filesOfKind("experiment").map((file) => ({ value: file.path }))}
            placeholder="main.py defaults"
          />
        </Field>
        <Field label="Search config" help="Overrides whatever the experiment config names.">
          <Select
            value={hparamConfig}
            onChange={setHparamConfig}
            options={filesOfKind("hparam").map((file) => ({ value: file.path }))}
            placeholder="from experiment config"
          />
        </Field>
        <Field label="Save dir" help="Relative to the repository root.">
          <TextInput value={saveDir} onChange={setSaveDir} mono placeholder="scratch/manual_run" />
        </Field>
        <Field label="Device">
          <Select
            value={device}
            onChange={setDevice}
            options={[{ value: "cuda" }, { value: "cpu" }]}
            placeholder="main.py default"
          />
        </Field>
        <Field label="SSL device" help="Where the semi-supervised step runs; blank follows the training device.">
          <Select
            value={sslDevice}
            onChange={setSslDevice}
            options={[{ value: "cuda" }, { value: "cpu" }]}
            placeholder="same as device"
          />
        </Field>
      </div>

      <div className="mt-5">
        <p className="studio-label mb-2">Extra arguments</p>
        <ArgsBuilder tokens={extraArgs} onChange={setExtraArgs} schema={schema} forbidden={[]} />
      </div>

      <div className="mt-5 space-y-3">
        <CommandPreview command={["python", "main.py", ...args]} />
        {error && <p className="text-xs text-[#ff9a9a]">{error}</p>}
        <Button variant="primary" onClick={launch} disabled={busy}>
          {busy ? <Loader2 className="size-3.5 animate-spin" /> : <PlayCircle className="size-3.5" />}
          Start run
        </Button>
      </div>
    </Card>
  );
}

function SchedulerPanel({ onLaunched }: { onLaunched: () => void }) {
  const { filesOfKind } = useStudio();
  const [manifest, setManifest] = useState<string | null>(null);
  const [device, setDevice] = useState<string | null>(null);
  const [gpus, setGpus] = useState("");
  const [maxConcurrent, setMaxConcurrent] = useState<number | null>(null);
  const [dryRun, setDryRun] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const args = useMemo(() => {
    const tokens: string[] = [];
    if (manifest) tokens.push("--manifest", manifest);
    if (device) tokens.push("--device", device);
    const gpuList = gpus.split(/[,\s]+/).filter(Boolean);
    if (gpuList.length > 0) tokens.push("--gpus", ...gpuList);
    if (maxConcurrent !== null) tokens.push("--max-concurrent-per-gpu", String(maxConcurrent));
    if (dryRun) tokens.push("--dry-run");
    return tokens;
  }, [manifest, device, gpus, maxConcurrent, dryRun]);

  const launch = async () => {
    setBusy(true);
    setError(null);
    try {
      await studioApi.launch({
        kind: "scheduler",
        label: `${manifest?.split("/").pop() ?? "manifest"}${dryRun ? " (dry run)" : ""}`,
        args,
        dryRun,
      });
      onLaunched();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card
      title="Scheduled batch"
      subtitle="Runs launch_runs.py, which places every enabled run in the manifest across the available devices."
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Run manifest">
          <Select
            value={manifest}
            onChange={setManifest}
            options={filesOfKind("manifest").map((file) => ({ value: file.path }))}
            placeholder="Pick a manifest…"
          />
        </Field>
        <Field label="Device override">
          <Select
            value={device}
            onChange={setDevice}
            options={[{ value: "cuda" }, { value: "cpu" }]}
            placeholder="from manifest"
          />
        </Field>
        <Field label="GPU override" help="Comma or space separated indexes/UUIDs.">
          <TextInput value={gpus} onChange={setGpus} mono placeholder="from manifest" />
        </Field>
        <Field label="Concurrent runs per GPU">
          <NumberInput value={maxConcurrent} onChange={setMaxConcurrent} placeholder="from manifest" />
        </Field>
      </div>

      <div className="mt-4">
        <Toggle
          checked={dryRun}
          onChange={setDryRun}
          label="Dry run"
          description="Print the placement wave the launcher would start, without starting anything."
        />
      </div>

      <div className="mt-5 space-y-3">
        <CommandPreview command={["python", "launch_runs.py", ...args]} />
        {error && <p className="text-xs text-[#ff9a9a]">{error}</p>}
        <Button variant="primary" onClick={launch} disabled={busy || !manifest}>
          {busy ? <Loader2 className="size-3.5 animate-spin" /> : <PlayCircle className="size-3.5" />}
          {dryRun ? "Preview placement" : "Start batch"}
        </Button>
      </div>
    </Card>
  );
}

function relativeTime(iso: string): string {
  const seconds = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000));
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86_400) return `${(seconds / 3600).toFixed(1)}h`;
  return `${(seconds / 86_400).toFixed(1)}d`;
}

function LogViewer({ launch }: { launch: LaunchStatus }) {
  const [text, setText] = useState("");
  const [follow, setFollow] = useState(true);
  const bodyRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const response = await studioApi.launchLog(launch.id);
        if (!cancelled) setText(response.text);
      } catch {
        // The log is created lazily; an early miss is not worth surfacing.
      }
    };
    void poll();
    const timer = setInterval(poll, 2000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [launch.id]);

  useEffect(() => {
    if (follow && bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  }, [text, follow]);

  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-3">
        <p className="truncate font-mono text-[0.7rem] text-[#8f9bb2]">{launch.logPath}</p>
        <Toggle checked={follow} onChange={setFollow} label="Follow" />
      </div>
      <pre
        ref={bodyRef}
        className="studio-scroll max-h-[28rem] overflow-auto rounded-xl border border-white/[0.08] bg-[#0a1120] p-3 font-mono text-[0.7rem] leading-relaxed text-[#b9c2d4]"
      >
        {text || "Waiting for output…"}
      </pre>
    </div>
  );
}

export function LaunchConsole() {
  const [mode, setMode] = useState<"main" | "scheduler">("scheduler");
  const [launches, setLaunches] = useState<LaunchStatus[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      const list = await studioApi.launches();
      setLaunches(list);
      setSelected((current) => current ?? list[0]?.id ?? null);
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = setInterval(refresh, 4000);
    return () => clearInterval(timer);
  }, [refresh]);

  const active = launches.find((launch) => launch.id === selected) ?? null;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap gap-2">
        {(
          [
            { id: "scheduler", label: "Scheduler", hint: "launch_runs.py" },
            { id: "main", label: "Single run", hint: "main.py" },
          ] as const
        ).map((option) => (
          <button
            key={option.id}
            type="button"
            onClick={() => setMode(option.id)}
            className={`focus-ring flex items-center gap-2 rounded-xl border px-3.5 py-2 text-xs transition-colors ${
              mode === option.id
                ? "border-[var(--violet)]/40 bg-[var(--violet)]/12 text-[#f7f8fc]"
                : "border-white/[0.1] text-[#b9c2d4] hover:bg-white/[0.04]"
            }`}
          >
            <Terminal className="size-3.5" />
            {option.label}
            <span className="font-mono text-[0.65rem] text-[#8f9bb2]">{option.hint}</span>
          </button>
        ))}
      </div>

      {mode === "main" ? (
        <SingleRunPanel onLaunched={refresh} />
      ) : (
        <SchedulerPanel onLaunched={refresh} />
      )}

      <Card
        title="Processes"
        subtitle="Started from this page and kept in logs/studio, so a dev-server reload does not lose them."
        actions={
          <>
            <Button onClick={refresh} disabled={refreshing}>
              <RefreshCw className={`size-3.5 ${refreshing ? "animate-spin" : ""}`} />
              Refresh
            </Button>
            <Button
              onClick={async () => {
                await studioApi.clearFinished();
                setSelected(null);
                await refresh();
              }}
            >
              <Eraser className="size-3.5" />
              Clear finished
            </Button>
          </>
        }
      >
        {launches.length === 0 ? (
          <p className="py-6 text-center text-xs text-[#8f9bb2]">Nothing launched yet.</p>
        ) : (
          <div className="studio-scroll -mx-1 overflow-x-auto px-1">
            <table className="w-full min-w-[42rem] text-left text-xs">
              <thead>
                <tr className="text-[0.65rem] uppercase tracking-wide text-[#8f9bb2]">
                  <th className="pb-2 pr-3 font-medium">Status</th>
                  <th className="pb-2 pr-3 font-medium">Label</th>
                  <th className="pb-2 pr-3 font-medium">Kind</th>
                  <th className="pb-2 pr-3 font-medium">PID</th>
                  <th className="pb-2 pr-3 font-medium">Started</th>
                  <th className="pb-2 font-medium" />
                </tr>
              </thead>
              <tbody>
                {launches.map((launch) => (
                  <tr
                    key={launch.id}
                    onClick={() => setSelected(launch.id)}
                    className={`cursor-pointer border-t border-white/[0.05] transition-colors ${
                      launch.id === selected ? "bg-[var(--violet)]/10" : "hover:bg-white/[0.03]"
                    }`}
                  >
                    <td className="py-2 pr-3">
                      <span className="flex items-center gap-1.5">
                        <Circle
                          className={`size-2 ${
                            launch.running
                              ? "fill-[var(--lime)] text-[var(--lime)]"
                              : launch.stoppedByUser
                                ? "fill-[#f5cf7a] text-[#f5cf7a]"
                                : "fill-[#8f9bb2] text-[#8f9bb2]"
                          }`}
                        />
                        {launch.running ? "running" : launch.stoppedByUser ? "stopped" : "finished"}
                      </span>
                    </td>
                    <td className="max-w-[16rem] truncate py-2 pr-3 font-mono text-[#e7ebf4]">
                      {launch.label}
                      {launch.dryRun && <span className="ml-2 text-[0.65rem] text-[#8f9bb2]">dry run</span>}
                    </td>
                    <td className="py-2 pr-3 text-[#8f9bb2]">{launch.kind}</td>
                    <td className="py-2 pr-3 font-mono text-[#8f9bb2]">{launch.pid}</td>
                    <td className="py-2 pr-3 text-[#8f9bb2]">{relativeTime(launch.startedAt)} ago</td>
                    <td className="py-2 text-right">
                      {launch.running && (
                        <button
                          type="button"
                          onClick={async (event) => {
                            event.stopPropagation();
                            await studioApi.stopLaunch(launch.id);
                            await refresh();
                          }}
                          className="focus-ring inline-flex items-center gap-1 rounded-lg border border-[#ff9a9a]/30 px-2 py-1 text-[0.7rem] text-[#ff9a9a] hover:bg-[#ff9a9a]/10"
                        >
                          <Square className="size-3" />
                          Stop
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {active && (
        <Card
          title={
            <span className="flex items-center gap-2">
              <ScrollText className="size-4 text-[var(--cyan)]" />
              {active.label}
            </span>
          }
          subtitle={active.command.join(" ")}
        >
          <LogViewer launch={active} />
        </Card>
      )}
    </div>
  );
}
