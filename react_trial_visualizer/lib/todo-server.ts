import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";

import type { TodoFile } from "@/lib/todo";

export const TODO_PATH = path.resolve(process.cwd(), "data", "todo.json");

/** Read at build time for the static page, and on every request by the API. */
export function loadTodoFile(): TodoFile | null {
  try {
    return JSON.parse(fs.readFileSync(TODO_PATH, "utf8")) as TodoFile;
  } catch {
    return null;
  }
}

// One write at a time: two quick checkbox clicks must not interleave read-modify-write.
let queue: Promise<unknown> = Promise.resolve();

export function updateTodoFile(change: (file: TodoFile) => void): Promise<TodoFile> {
  const next = queue.then(async () => {
    const file = JSON.parse(await fsp.readFile(TODO_PATH, "utf8")) as TodoFile;
    change(file);
    file.updatedAt = new Date().toISOString().slice(0, 10);
    const temporary = `${TODO_PATH}.tmp`;
    await fsp.writeFile(temporary, `${JSON.stringify(file, null, 2)}\n`, "utf8");
    await fsp.rename(temporary, TODO_PATH);
    return file;
  });
  queue = next.catch(() => undefined);
  return next;
}
