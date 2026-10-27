/** The Phase 2 todo list: data/todo.json, read by /todo/ and updated through /api/todo/. */

export interface TodoLink {
  label: string;
  href: string;
}

export interface TodoItem {
  id: string;
  group: string;
  rank: number;
  plan: string;
  title: string;
  why: string;
  effort: string;
  steps: string[];
  commands: string[];
  links: TodoLink[];
  done: boolean;
  doneAt?: string | null;
  note: string;
  /** Added from the page rather than from the plan; only these can be deleted there. */
  custom?: boolean;
}

export interface TodoGroup {
  id: string;
  title: string;
  note: string;
}

export interface TodoFile {
  version: number;
  title: string;
  subtitle: string;
  updatedAt: string;
  groups: TodoGroup[];
  items: TodoItem[];
}

/** A change sent to PATCH /api/todo/. */
export interface TodoPatch {
  id: string;
  done?: boolean;
  note?: string;
  group?: string;
}
