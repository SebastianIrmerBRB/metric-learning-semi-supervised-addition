import { NextResponse } from "next/server";

import type { TodoItem, TodoPatch } from "@/lib/todo";
import { loadTodoFile, updateTodoFile } from "@/lib/todo-server";

export const dynamic = "force-dynamic";

function fail(message: string, status = 400) {
  return NextResponse.json({ error: message }, { status });
}

/** The whole list, as data/todo.json holds it now. */
export async function GET() {
  const file = loadTodoFile();
  return file ? NextResponse.json(file) : fail("data/todo.json is missing or not valid JSON.", 404);
}

/** Check an item off, or change its note or group: `{ id, done?, note?, group? }`. */
export async function PATCH(request: Request) {
  const patch = (await request.json().catch(() => null)) as TodoPatch | null;
  if (!patch?.id) return fail("Send { id, done?, note?, group? }.");
  try {
    const file = await updateTodoFile((current) => {
      const item = current.items.find((candidate) => candidate.id === patch.id);
      if (!item) throw new Error(`No item ${patch.id}.`);
      if (typeof patch.done === "boolean" && patch.done !== item.done) {
        item.done = patch.done;
        item.doneAt = patch.done ? new Date().toISOString() : null;
      }
      if (typeof patch.note === "string") item.note = patch.note.slice(0, 4000);
      if (typeof patch.group === "string" && current.groups.some((group) => group.id === patch.group)) item.group = patch.group;
    });
    return NextResponse.json(file);
  } catch (error) {
    return fail(error instanceof Error ? error.message : String(error));
  }
}

/** Add an item of your own: `{ title, group?, why? }`. It goes to the end of its group. */
export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as { title?: string; group?: string; why?: string } | null;
  const title = body?.title?.trim();
  if (!title) return fail("Send { title, group?, why? }.");
  const file = await updateTodoFile((current) => {
    const group = current.groups.some((candidate) => candidate.id === body?.group) ? body!.group! : current.groups[0].id;
    const item: TodoItem = {
      id: `custom-${Date.now().toString(36)}`,
      group,
      rank: Math.max(0, ...current.items.map((candidate) => candidate.rank)) + 1,
      plan: "",
      title: title.slice(0, 300),
      why: (body?.why ?? "").slice(0, 2000),
      effort: "",
      steps: [],
      commands: [],
      links: [],
      done: false,
      note: "",
      custom: true,
    };
    current.items.push(item);
  });
  return NextResponse.json(file);
}

/** Remove an item added from the page (`?id=`); items from the plan cannot be deleted here. */
export async function DELETE(request: Request) {
  const id = new URL(request.url).searchParams.get("id");
  try {
    const file = await updateTodoFile((current) => {
      const item = current.items.find((candidate) => candidate.id === id);
      if (!item) throw new Error(`No item ${id}.`);
      if (!item.custom) throw new Error("Only items added on this page can be deleted; check plan items off instead.");
      current.items = current.items.filter((candidate) => candidate.id !== id);
    });
    return NextResponse.json(file);
  } catch (error) {
    return fail(error instanceof Error ? error.message : String(error));
  }
}
