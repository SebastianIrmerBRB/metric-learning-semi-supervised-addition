import type { Metadata } from "next";
import { TodoBoard } from "@/components/TodoBoard";
import { loadTodoFile } from "@/lib/todo-server";

export const dynamic = "force-static";

export const metadata: Metadata = {
  title: "Phase 2 todo · Trial Atlas",
  description: "What to do next for Phase 2, ranked, with links to the page where each task is done.",
};

export default function TodoPage() {
  return <TodoBoard initial={loadTodoFile()} />;
}
