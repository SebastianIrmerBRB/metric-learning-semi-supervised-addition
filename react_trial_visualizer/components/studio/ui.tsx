"use client";

import {
  AlertTriangle,
  Check,
  ChevronDown,
  ChevronRight,
  Info,
  Plus,
  X,
  XCircle,
} from "lucide-react";
import {
  useId,
  useMemo,
  useState,
  type ChangeEvent,
  type ReactNode,
} from "react";

import type { Issue, IssueLevel } from "@/lib/studio/schema";

/* -------------------------------------------------------------- containers */

export function Card({
  title,
  subtitle,
  actions,
  children,
  tone = "default",
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  tone?: "default" | "soft";
}) {
  return (
    <section className={tone === "soft" ? "surface-soft p-4" : "surface p-5"}>
      {(title || actions) && (
        <header className="mb-4 flex items-start justify-between gap-4">
          <div className="min-w-0">
            {title && <h2 className="text-sm font-semibold text-[#f7f8fc]">{title}</h2>}
            {subtitle && <p className="mt-1 text-xs leading-relaxed text-[#8f9bb2]">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Collapsible({
  title,
  subtitle,
  badge,
  defaultOpen = false,
  children,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  badge?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="surface-soft overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="focus-ring flex w-full items-center gap-3 px-4 py-3 text-left"
      >
        {open ? (
          <ChevronDown className="size-4 shrink-0 text-[#8f9bb2]" />
        ) : (
          <ChevronRight className="size-4 shrink-0 text-[#8f9bb2]" />
        )}
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium text-[#e7ebf4]">{title}</span>
          {subtitle && <span className="block truncate text-xs text-[#8f9bb2]">{subtitle}</span>}
        </span>
        {badge}
      </button>
      {open && <div className="border-t border-white/[0.06] p-4">{children}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ fields */

export function Field({
  label,
  help,
  hint,
  children,
  issues,
  wide,
}: {
  label: string;
  help?: string;
  hint?: ReactNode;
  children: ReactNode;
  issues?: Issue[];
  wide?: boolean;
}) {
  const relevant = issues ?? [];
  return (
    <div className={wide ? "sm:col-span-2" : undefined}>
      <label className="studio-label mb-1.5">{label}</label>
      {children}
      {help && <p className="mt-1 text-[0.7rem] leading-snug text-[#8f9bb2]">{help}</p>}
      {hint && <div className="mt-1 text-[0.7rem] leading-snug text-[#8f9bb2]">{hint}</div>}
      {relevant.length > 0 && (
        <ul className="mt-1.5 space-y-1">
          {relevant.map((issue, index) => (
            <li key={index} className={`flex gap-1.5 text-[0.7rem] leading-snug ${issueTextClass(issue.level)}`}>
              <IssueIcon level={issue.level} />
              <span>{issue.message}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function TextInput({
  value,
  onChange,
  placeholder,
  invalid,
  disabled,
  mono,
  list,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  invalid?: boolean;
  disabled?: boolean;
  mono?: boolean;
  list?: string;
}) {
  return (
    <input
      className={`studio-input ${invalid ? "is-invalid" : ""} ${mono ? "font-mono text-xs" : ""}`}
      value={value}
      list={list}
      disabled={disabled}
      placeholder={placeholder}
      onChange={(event: ChangeEvent<HTMLInputElement>) => onChange(event.target.value)}
    />
  );
}

export function NumberInput({
  value,
  onChange,
  placeholder,
  step,
  disabled,
  invalid,
}: {
  value: number | null | undefined;
  onChange: (value: number | null) => void;
  placeholder?: string;
  step?: string;
  disabled?: boolean;
  invalid?: boolean;
}) {
  // Kept as text so scientific notation such as 1e-5 survives typing.
  const [draft, setDraft] = useState<string | null>(null);
  const shown = draft ?? (value === null || value === undefined ? "" : String(value));
  return (
    <input
      className={`studio-input ${invalid ? "is-invalid" : ""}`}
      inputMode="decimal"
      step={step}
      disabled={disabled}
      placeholder={placeholder}
      value={shown}
      onChange={(event) => {
        const text = event.target.value;
        setDraft(text);
        if (text.trim() === "") {
          onChange(null);
          return;
        }
        const parsed = Number(text);
        if (Number.isFinite(parsed)) onChange(parsed);
      }}
      onBlur={() => setDraft(null)}
    />
  );
}

export function Select({
  value,
  onChange,
  options,
  placeholder = "—",
  disabled,
  invalid,
  allowEmpty = true,
}: {
  value: string | null | undefined;
  onChange: (value: string | null) => void;
  options: { value: string; label?: string }[];
  placeholder?: string;
  disabled?: boolean;
  invalid?: boolean;
  allowEmpty?: boolean;
}) {
  const current = value ?? "";
  const missing = current !== "" && !options.some((option) => option.value === current);
  return (
    <select
      className={`studio-input ${invalid || missing ? "is-invalid" : ""}`}
      value={current}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value === "" ? null : event.target.value)}
    >
      {(allowEmpty || current === "") && <option value="">{placeholder}</option>}
      {missing && <option value={current}>{current} (not found)</option>}
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label ?? option.value}
        </option>
      ))}
    </select>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  description,
  disabled,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  label: string;
  description?: string;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className="flex items-start gap-3">
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={`focus-ring relative mt-0.5 h-5 w-9 shrink-0 rounded-full border transition-colors ${
          checked
            ? "border-transparent bg-[var(--violet)]"
            : "border-white/[0.12] bg-white/[0.06]"
        } ${disabled ? "opacity-50" : ""}`}
      >
        <span
          className={`absolute top-0.5 size-3.5 rounded-full bg-white transition-transform ${
            checked ? "translate-x-4" : "translate-x-0.5"
          }`}
        />
      </button>
      <label htmlFor={id} className="min-w-0">
        <span className="block text-xs font-medium text-[#e7ebf4]">{label}</span>
        {description && <span className="block text-[0.7rem] leading-snug text-[#8f9bb2]">{description}</span>}
      </label>
    </div>
  );
}

/** A text input backed by a datalist, so known values are one keystroke away. */
export function Combobox({
  value,
  onChange,
  onCommit,
  options,
  placeholder,
  invalid,
  mono = true,
}: {
  value: string;
  onChange: (value: string) => void;
  /** Called when the field is left or Enter is pressed, for deferred renames. */
  onCommit?: () => void;
  options: { value: string; label?: string }[];
  placeholder?: string;
  invalid?: boolean;
  mono?: boolean;
}) {
  const id = useId();
  return (
    <>
      <input
        className={`studio-input ${invalid ? "is-invalid" : ""} ${mono ? "font-mono text-xs" : ""}`}
        value={value}
        list={id}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
        onBlur={onCommit}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            onCommit?.();
          }
        }}
      />
      <datalist id={id}>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </datalist>
    </>
  );
}

/** An editable list of scalars rendered as removable chips. */
export function ChipList({
  values,
  onChange,
  placeholder = "Add value…",
  suggestions = [],
  parse = (text: string) => text,
}: {
  values: unknown[];
  onChange: (values: unknown[]) => void;
  placeholder?: string;
  suggestions?: string[];
  parse?: (text: string) => unknown;
}) {
  const [draft, setDraft] = useState("");
  const listId = useId();

  const commit = () => {
    const text = draft.trim();
    if (!text) return;
    onChange([...values, parse(text)]);
    setDraft("");
  };

  return (
    <div>
      <div className="flex flex-wrap items-center gap-1.5">
        {values.map((item, index) => (
          <span key={index} className="studio-chip">
            {typeof item === "string" ? item : JSON.stringify(item)}
            <button
              type="button"
              aria-label="Remove"
              onClick={() => onChange(values.filter((_, position) => position !== index))}
              className="focus-ring -mr-1 rounded-full p-0.5 text-[#8f9bb2] hover:text-[#f7f8fc]"
            >
              <X className="size-3" />
            </button>
          </span>
        ))}
      </div>
      <div className="mt-2 flex gap-1.5">
        <input
          className="studio-input font-mono text-xs"
          value={draft}
          list={listId}
          placeholder={placeholder}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              commit();
            }
          }}
        />
        <datalist id={listId}>
          {suggestions.map((item) => (
            <option key={item} value={item} />
          ))}
        </datalist>
        <button
          type="button"
          onClick={commit}
          className="focus-ring flex h-[2.35rem] shrink-0 items-center gap-1 rounded-[0.7rem] border border-white/[0.12] px-3 text-xs text-[#d9e0ed] hover:bg-white/[0.05]"
        >
          <Plus className="size-3.5" />
          Add
        </button>
      </div>
    </div>
  );
}

/** Free-form string→scalar mapping, used for sampler_params and env blocks. */
export function KeyValueEditor({
  entries,
  onChange,
  keyPlaceholder = "key",
  valuePlaceholder = "value",
}: {
  entries: Record<string, unknown>;
  onChange: (entries: Record<string, unknown>) => void;
  keyPlaceholder?: string;
  valuePlaceholder?: string;
}) {
  const [draftKey, setDraftKey] = useState("");
  const [draftValue, setDraftValue] = useState("");
  // Typing "1.5" passes through "1.", which would round-trip to 1 and eat the
  // decimal point, so the raw text stays authoritative until the field blurs.
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const rows = Object.entries(entries);

  return (
    <div className="space-y-1.5">
      {rows.map(([key, value]) => (
        <div key={key} className="flex items-center gap-1.5">
          <span className="studio-chip h-[2.35rem] min-w-[9rem] justify-start rounded-[0.7rem]">{key}</span>
          <input
            className="studio-input font-mono text-xs"
            value={drafts[key] ?? (typeof value === "string" ? value : JSON.stringify(value))}
            onChange={(event) => {
              const text = event.target.value;
              setDrafts((current) => ({ ...current, [key]: text }));
              onChange({ ...entries, [key]: parseScalar(text) });
            }}
            onBlur={() =>
              setDrafts((current) => {
                const next = { ...current };
                delete next[key];
                return next;
              })
            }
          />
          <button
            type="button"
            aria-label={`Remove ${key}`}
            onClick={() => {
              const next = { ...entries };
              delete next[key];
              onChange(next);
            }}
            className="focus-ring rounded-lg p-2 text-[#8f9bb2] hover:text-[#f7f8fc]"
          >
            <X className="size-3.5" />
          </button>
        </div>
      ))}
      <div className="flex items-center gap-1.5">
        <input
          className="studio-input min-w-[9rem] font-mono text-xs"
          value={draftKey}
          placeholder={keyPlaceholder}
          onChange={(event) => setDraftKey(event.target.value)}
        />
        <input
          className="studio-input font-mono text-xs"
          value={draftValue}
          placeholder={valuePlaceholder}
          onChange={(event) => setDraftValue(event.target.value)}
        />
        <button
          type="button"
          aria-label="Add entry"
          disabled={!draftKey.trim()}
          onClick={() => {
            onChange({ ...entries, [draftKey.trim()]: parseScalar(draftValue) });
            setDraftKey("");
            setDraftValue("");
          }}
          className="focus-ring rounded-lg p-2 text-[#8f9bb2] hover:text-[#f7f8fc] disabled:opacity-40"
        >
          <Plus className="size-3.5" />
        </button>
      </div>
    </div>
  );
}

/** Turn typed text into the JSON scalar it obviously represents. */
export function parseScalar(raw: string): unknown {
  const text = raw.trim();
  if (text === "") return "";
  if (text === "true") return true;
  if (text === "false") return false;
  if (text === "null") return null;
  const asNumber = Number(text);
  if (Number.isFinite(asNumber) && /^-?\d/.test(text)) return asNumber;
  return raw;
}

/* ------------------------------------------------------------------ issues */

export function issueTextClass(level: IssueLevel): string {
  if (level === "error") return "text-[#ff9a9a]";
  if (level === "warning") return "text-[#f5cf7a]";
  return "text-[#8ab4d8]";
}

export function IssueIcon({ level }: { level: IssueLevel }) {
  const className = "mt-[0.15rem] size-3 shrink-0";
  if (level === "error") return <XCircle className={className} />;
  if (level === "warning") return <AlertTriangle className={className} />;
  return <Info className={className} />;
}

export function IssueList({ issues, empty }: { issues: Issue[]; empty?: ReactNode }) {
  const ordered = useMemo(() => {
    const rank: Record<IssueLevel, number> = { error: 0, warning: 1, info: 2 };
    return [...issues].sort((left, right) => rank[left.level] - rank[right.level]);
  }, [issues]);

  if (ordered.length === 0) {
    return (
      <p className="flex items-center gap-2 text-xs text-[#b8e56e]">
        <Check className="size-3.5" />
        {empty ?? "No problems found."}
      </p>
    );
  }

  return (
    <ul className="space-y-1.5">
      {ordered.map((issue, index) => (
        <li key={index} className={`flex gap-2 text-xs leading-snug ${issueTextClass(issue.level)}`}>
          <IssueIcon level={issue.level} />
          <span>
            {issue.field && <span className="font-mono text-[0.7rem] opacity-80">{issue.field} · </span>}
            {issue.message}
          </span>
        </li>
      ))}
    </ul>
  );
}

export function IssueBadge({ issues }: { issues: Issue[] }) {
  const errors = issues.filter((issue) => issue.level === "error").length;
  const warnings = issues.filter((issue) => issue.level === "warning").length;
  if (errors === 0 && warnings === 0) return null;
  return (
    <span className="flex shrink-0 items-center gap-1.5 text-[0.7rem]">
      {errors > 0 && (
        <span className="rounded-full bg-[#ff9a9a]/15 px-2 py-0.5 font-medium text-[#ff9a9a]">{errors}</span>
      )}
      {warnings > 0 && (
        <span className="rounded-full bg-[#f5cf7a]/15 px-2 py-0.5 font-medium text-[#f5cf7a]">{warnings}</span>
      )}
    </span>
  );
}

/* ----------------------------------------------------------------- buttons */

export function Button({
  children,
  onClick,
  variant = "ghost",
  disabled,
  title,
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost" | "danger";
  disabled?: boolean;
  title?: string;
  type?: "button" | "submit";
}) {
  const styles = {
    primary:
      "border-transparent bg-[var(--violet)] text-[#0b1020] hover:bg-[#ab9dff] font-semibold",
    ghost: "border-white/[0.12] text-[#d9e0ed] hover:bg-white/[0.05]",
    danger: "border-[#ff9a9a]/30 text-[#ff9a9a] hover:bg-[#ff9a9a]/10",
  }[variant];

  return (
    <button
      type={type}
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={`focus-ring inline-flex h-9 items-center gap-1.5 rounded-xl border px-3 text-xs transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${styles}`}
    >
      {children}
    </button>
  );
}
