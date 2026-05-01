// frontend/components/chat/SessionsSidebar.tsx
"use client";
import { Fragment, useEffect, useRef, useState } from "react";
import {
  Plus, Pin, PinOff, Pencil, Trash2, MoreHorizontal,
  PanelLeftClose, PanelLeftOpen, LogOut, ChevronDown,
  Bug,
} from "lucide-react";
import { api, type SessionListItem } from "@/lib/api";
import type { SessionState } from "@/lib/api-types";

const COLLAPSE_KEY = "socratic-ot.sidebar_collapsed";

/** Recent-chats sidebar with a collapse toggle.
 *
 *  States:
 *    - expanded (~18rem) — full chat list, weak-topics card under New chat
 *    - collapsed (~3.5rem) — slim rail with new-chat icon + expand button
 *
 *  Collapse state persists in localStorage so reload preserves the user's
 *  preference. Parent owns the active session (via useSession).
 */
export function SessionsSidebar({
  activeId,
  refreshKey,
  onSelect,
  onNew,
  weakTopics,
  debugInfo,
}: {
  activeId: string | null;
  refreshKey: number;
  onSelect: (id: string) => void;
  onNew:    () => Promise<string | void>;
  weakTopics: string[];
  /** Optional — when present, renders a collapsible debug card showing
   *  turn count, current concept, last classifier label, CRAG decision,
   *  IDK count, etc. Pass null to hide the card. */
  debugInfo?: SessionState | null;
}) {
  const [sessions, setSessions] = useState<SessionListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(false);

  // Hydrate collapse state from localStorage on mount.
  useEffect(() => {
    if (typeof window === "undefined") return;
    setCollapsed(window.localStorage.getItem(COLLAPSE_KEY) === "1");
  }, []);

  function toggleCollapsed() {
    const next = !collapsed;
    setCollapsed(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(COLLAPSE_KEY, next ? "1" : "0");
    }
  }

  async function reload() {
    try {
      const r = await api.listSessions();
      setSessions(r.sessions);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "couldn't load");
      setSessions([]);
    }
  }

  useEffect(() => { reload(); }, [refreshKey]);

  // Close any open three-dot menu on outside click.
  useEffect(() => {
    function onDoc() { setOpenMenuId(null); }
    if (openMenuId) {
      document.addEventListener("click", onDoc);
      return () => document.removeEventListener("click", onDoc);
    }
  }, [openMenuId]);

  async function handlePin(s: SessionListItem) {
    setOpenMenuId(null);
    await api.patchSession(s.id, { pinned: !s.pinned });
    reload();
  }
  async function handleDelete(s: SessionListItem) {
    setOpenMenuId(null);
    const label = displayTitle(s);
    if (!window.confirm(`Delete "${label}"? This can't be undone.`)) return;
    await api.deleteSession(s.id);
    if (s.id === activeId) {
      const r = await api.listSessions();
      const next = r.sessions[0]?.id;
      if (next) onSelect(next);
      else      await onNew();
    }
    reload();
  }
  function startRename(s: SessionListItem) {
    setOpenMenuId(null);
    setRenamingId(s.id);
  }
  async function commitRename(id: string, value: string) {
    setRenamingId(null);
    const trimmed = value.trim();
    if (!trimmed) return;
    await api.patchSession(id, { title: trimmed });
    reload();
  }

  const pinned = sessions?.filter((s) => s.pinned)  ?? [];
  const recent = sessions?.filter((s) => !s.pinned) ?? [];
  const weakCount = weakTopics?.length ?? 0;

  // ── Collapsed rail ────────────────────────────────────────────────────────
  if (collapsed) {
    return (
      <aside className="flex w-14 shrink-0 flex-col items-center gap-2 py-1">
        <IconButton
          label="Expand sidebar"
          onClick={toggleCollapsed}
        >
          <PanelLeftOpen size={18} />
        </IconButton>

        <IconButton
          label="New chat"
          onClick={async () => {
            const id = await onNew();
            if (typeof id === "string") onSelect(id);
            setTimeout(reload, 80);
          }}
          accent
        >
          <Plus size={18} />
        </IconButton>

        {weakCount > 0 && (
          <button
            type="button"
            onClick={toggleCollapsed}
            title={`${weakCount} weak topic${weakCount === 1 ? "" : "s"} — click to expand`}
            className="relative mt-1 grid h-9 w-9 place-items-center rounded-full border border-coral-200 bg-coral-50 text-[11px] font-medium text-coral-700 ring-focus"
            aria-label={`${weakCount} weak topics`}
          >
            {weakCount}
          </button>
        )}
      </aside>
    );
  }

  // ── Expanded rail ─────────────────────────────────────────────────────────
  // Layout: a column that fills the available height. The chats list is
  // the only flex-1 / scrollable section so:
  //   - New chat button stays pinned at the top
  //   - Weak-topics card stays visible
  //   - Many chats → list scrolls; the rest of the sidebar doesn't grow
  //   - Logout button stays pinned at the bottom
  return (
    <aside className="flex w-full shrink-0 flex-col gap-3 md:w-72"
           style={{ height: "calc(100vh - 5.5rem)" }}>
      {/* Header — New chat + collapse toggle (fixed at top) */}
      <div className="flex shrink-0 items-center justify-between gap-2">
        <button
          type="button"
          onClick={async () => {
            const id = await onNew();
            if (typeof id === "string") onSelect(id);
            setTimeout(reload, 80);
          }}
          className="flex flex-1 items-center justify-center gap-2 rounded-2xl border border-ivory-200 bg-white px-3 py-2 text-sm font-medium text-ink shadow-sm transition hover:border-coral-300 hover:bg-coral-50 hover:text-coral-800 ring-focus"
        >
          <Plus size={16} />
          New chat
        </button>
        <IconButton label="Collapse sidebar" onClick={toggleCollapsed}>
          <PanelLeftClose size={16} />
        </IconButton>
      </div>

      {/* Weak topics — collapsed by default to 3, click header to expand */}
      <div className="shrink-0">
        <WeakTopicsCard topics={weakTopics} />
      </div>

      {/* Debug — graph state for the active session, collapsible */}
      {debugInfo && (
        <div className="shrink-0">
          <DebugCard info={debugInfo} />
        </div>
      )}

      {error && (
        <div className="shrink-0 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          Couldn’t load chats: {error}
        </div>
      )}

      {/* Chats list — flex-1 + overflow-y-auto so it absorbs all leftover
          height and scrolls internally instead of pushing the logout
          button off-screen. */}
      <div className="-mr-1 min-h-0 flex-1 overflow-y-auto pr-1">
        {pinned.length > 0 && (
          <Section label="Pinned">
            {pinned.map((s) => (
              <Row
                key={s.id}
                s={s}
                active={s.id === activeId}
                menuOpen={openMenuId === s.id}
                renaming={renamingId === s.id}
                onSelect={() => onSelect(s.id)}
                onMenu={(e) => { e.stopPropagation(); setOpenMenuId(openMenuId === s.id ? null : s.id); }}
                onPin={() => handlePin(s)}
                onRename={() => startRename(s)}
                onDelete={() => handleDelete(s)}
                onRenameCommit={(v) => commitRename(s.id, v)}
                onRenameCancel={() => setRenamingId(null)}
              />
            ))}
          </Section>
        )}

        <Section label={pinned.length > 0 ? "Recent" : "Chats"}>
          {sessions === null ? (
            <SkeletonRows />
          ) : recent.length === 0 ? (
            <div className="px-3 py-3 text-xs text-ivory-500">
              {pinned.length > 0 ? "No other chats yet." : "No chats yet."}
            </div>
          ) : (
            recent.map((s) => (
              <Row
                key={s.id}
                s={s}
                active={s.id === activeId}
                menuOpen={openMenuId === s.id}
                renaming={renamingId === s.id}
                onSelect={() => onSelect(s.id)}
                onMenu={(e) => { e.stopPropagation(); setOpenMenuId(openMenuId === s.id ? null : s.id); }}
                onPin={() => handlePin(s)}
                onRename={() => startRename(s)}
                onDelete={() => handleDelete(s)}
                onRenameCommit={(v) => commitRename(s.id, v)}
                onRenameCancel={() => setRenamingId(null)}
              />
            ))
          )}
        </Section>
      </div>

      {/* Logout — pinned at the bottom. Stub for now; wires to /login
          once auth ships. */}
      <LogoutButton />
    </aside>
  );
}

function LogoutButton() {
  function handleLogout() {
    // Auth doesn't exist yet — show a neutral placeholder so the
    // button has visible behavior. Once /login ships, replace this
    // with a real signout: clear session_id + user_id from
    // localStorage and router.push("/login").
    if (typeof window === "undefined") return;
    const ok = window.confirm(
      "Login isn't wired up yet. For now this just clears your local session. Continue?",
    );
    if (!ok) return;
    window.localStorage.removeItem("socratic-ot.session_id");
    // Intentionally NOT clearing user_id — that's the cross-session
    // memory anchor; tying its lifetime to auth comes later.
    window.location.reload();
  }
  return (
    <button
      type="button"
      onClick={handleLogout}
      className="flex shrink-0 items-center justify-center gap-2 rounded-2xl border border-ivory-200 bg-white px-3 py-2 text-sm text-ivory-600 shadow-sm transition hover:border-rose-200 hover:bg-rose-50 hover:text-rose-700 ring-focus"
      title="Sign out (coming soon)"
    >
      <LogOut size={15} />
      <span>Logout</span>
    </button>
  );
}

const WEAK_TOPICS_COLLAPSED = 3;

function WeakTopicsCard({ topics }: { topics: string[] }) {
  const [expanded, setExpanded] = useState(false);
  const total = topics.length;
  const visible = expanded ? topics : topics.slice(0, WEAK_TOPICS_COLLAPSED);
  const hasOverflow = total > WEAK_TOPICS_COLLAPSED;

  // Make the whole header clickable when there's overflow, so the user
  // doesn't have to aim for a tiny chevron.
  const headerInteractive = hasOverflow;

  return (
    <div className="rounded-2xl border border-ivory-200 bg-white p-3 shadow-sm">
      <button
        type="button"
        disabled={!headerInteractive}
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
        aria-controls="weak-topics-list"
        className={`flex w-full items-center justify-between text-left ${
          headerInteractive ? "cursor-pointer" : "cursor-default"
        }`}
      >
        <span className="text-[10.5px] font-medium uppercase tracking-[0.08em] text-ivory-500">
          Weak topics{total > 0 && ` · ${total}`}
        </span>
        {hasOverflow && (
          <ChevronDown
            size={14}
            className={`text-ivory-500 transition-transform ${
              expanded ? "rotate-180" : ""
            }`}
            aria-hidden
          />
        )}
      </button>

      <div id="weak-topics-list" className="mt-1.5">
        {total === 0 ? (
          <div className="text-xs text-ivory-400">none yet</div>
        ) : (
          <>
            <ul className="flex flex-wrap gap-1.5">
              {visible.map((t) => (
                <li
                  key={t}
                  className="rounded-full border border-coral-200 bg-coral-50 px-2.5 py-0.5 text-[11.5px] text-coral-800"
                  title={t}
                >
                  {t}
                </li>
              ))}
            </ul>
            {hasOverflow && !expanded && (
              <button
                type="button"
                onClick={() => setExpanded(true)}
                className="mt-1.5 text-[11px] text-coral-700 hover:underline"
              >
                +{total - WEAK_TOPICS_COLLAPSED} more · show all
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}

/** Compact graph-state inspector for the active session. Collapsed by
 *  default — click the header to reveal. Each field maps directly to
 *  one key on the SessionState payload returned by GET /sessions/{id};
 *  empty values render as a dim `—` so it's obvious when the graph
 *  hasn't populated something yet (e.g. before the first turn). */
function DebugCard({ info }: { info: SessionState }) {
  const [open, setOpen] = useState(false);
  const sid = info.session_id ? info.session_id.slice(-8) : "—";
  const yn = (b?: boolean) => (b ? "yes" : "no");
  const dash = (s?: string | number) =>
    s === "" || s === undefined || s === null ? "—" : String(s);

  const rows: Array<[string, string]> = [
    ["turn",        dash(info.turn_count)],
    ["mode",        dash(info.mode)],
    ["phase",       dash(info.student_phase)],
    ["concept",     dash(info.current_concept)],
    ["mastered",    yn(info.concept_mastered)],
    ["mastery",     dash(info.mastery_level)],
    ["idk_count",   dash(info.idk_count)],
    ["attempted",   yn(info.student_attempted)],
    ["classifier",  dash(info.classifier_output)],
    ["crag",        dash(info.crag_decision)],
    ["src_node",    dash(info.draft_source_node)],
    ["topic_chc",   dash(info.topic_choice)],
    ["mastery_chc", dash(info.mastery_choice)],
    ["dean_revs",   dash(info.dean_revisions)],
    ["session",     sid],
  ];

  return (
    <div className="rounded-2xl border border-ivory-200 bg-white p-3 shadow-sm">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-controls="debug-card-body"
        className="flex w-full items-center justify-between text-left"
      >
        <span className="inline-flex items-center gap-1.5 text-[10.5px] font-medium uppercase tracking-[0.08em] text-ivory-500">
          <Bug size={12} aria-hidden /> Debug
        </span>
        <ChevronDown
          size={14}
          className={`text-ivory-500 transition-transform ${open ? "rotate-180" : ""}`}
          aria-hidden
        />
      </button>

      {open && (
        <dl
          id="debug-card-body"
          className="mt-2 grid grid-cols-[6.5rem_1fr] gap-x-2 gap-y-1 font-mono text-[10.5px] leading-snug"
        >
          {rows.map(([k, v]) => (
            <Fragment key={k}>
              <dt className="text-ivory-500">{k}</dt>
              <dd
                className={`truncate ${
                  v === "—" ? "text-ivory-400" : "text-ink"
                }`}
                title={v}
              >
                {v}
              </dd>
            </Fragment>
          ))}
        </dl>
      )}
    </div>
  );
}


function IconButton({
  children, label, onClick, accent = false,
}: {
  children: React.ReactNode;
  label:    string;
  onClick:  () => void;
  accent?:  boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className={`grid h-9 w-9 place-items-center rounded-xl border border-ivory-200 transition ring-focus ${
        accent
          ? "bg-ink text-ivory-50 hover:bg-ivory-700 border-transparent shadow-sm"
          : "bg-white text-ivory-600 hover:bg-ivory-100 hover:text-ink"
      }`}
    >
      {children}
    </button>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1 px-2 text-[10.5px] font-medium uppercase tracking-[0.08em] text-ivory-500">
        {label}
      </div>
      <ul className="flex flex-col">{children}</ul>
    </div>
  );
}

function SkeletonRows() {
  return (
    <>
      {[0, 1, 2].map((i) => (
        <li key={i} className="my-0.5 h-9 animate-pulse rounded-lg bg-ivory-100/70" />
      ))}
    </>
  );
}

function Row({
  s, active, menuOpen, renaming,
  onSelect, onMenu, onPin, onRename, onDelete,
  onRenameCommit, onRenameCancel,
}: {
  s: SessionListItem;
  active: boolean;
  menuOpen: boolean;
  renaming: boolean;
  onSelect: () => void;
  onMenu:   (e: React.MouseEvent) => void;
  onPin:    () => void;
  onRename: () => void;
  onDelete: () => void;
  onRenameCommit: (value: string) => void;
  onRenameCancel: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (renaming) inputRef.current?.select();
  }, [renaming]);

  return (
    <li
      className={`group relative my-0.5 flex items-center gap-2 rounded-lg px-2 py-1.5 transition-colors ${
        active
          ? "bg-ivory-100 text-ink"
          : "text-ivory-700 hover:bg-ivory-100"
      }`}
    >
      {renaming ? (
        <input
          ref={inputRef}
          defaultValue={displayTitle(s)}
          onKeyDown={(e) => {
            if (e.key === "Enter") onRenameCommit((e.target as HTMLInputElement).value);
            if (e.key === "Escape") onRenameCancel();
          }}
          onBlur={(e) => onRenameCommit(e.target.value)}
          className="min-w-0 flex-1 rounded border border-coral-300 bg-white px-1.5 py-0.5 text-[13px] text-ink outline-none focus:border-coral-500"
        />
      ) : (
        <button
          type="button"
          onClick={onSelect}
          className="flex min-w-0 flex-1 items-center gap-1.5 text-left"
          title={displayTitle(s)}
        >
          {s.pinned && (
            <Pin size={12} className="shrink-0 text-coral-500" aria-label="pinned" />
          )}
          <span className="truncate text-[13px] leading-5">
            {displayTitle(s)}
          </span>
        </button>
      )}

      {!renaming && (
        <button
          type="button"
          onClick={onMenu}
          aria-label="Chat options"
          className={`grid h-7 w-7 shrink-0 place-items-center rounded text-ivory-500 transition ${
            menuOpen ? "bg-ivory-200 text-ink" : "opacity-0 group-hover:opacity-100 hover:bg-ivory-200 hover:text-ink"
          } ${active ? "opacity-100" : ""}`}
        >
          <MoreHorizontal size={15} />
        </button>
      )}

      {menuOpen && (
        <div
          onClick={(e) => e.stopPropagation()}
          className="absolute right-1 top-9 z-20 w-40 rounded-lg border border-ivory-200 bg-white py-1 text-[13px] shadow-md"
        >
          <MenuItem icon={s.pinned ? <PinOff size={14} /> : <Pin size={14} />}
                    label={s.pinned ? "Unpin" : "Pin"}
                    onClick={onPin} />
          <MenuItem icon={<Pencil size={14} />} label="Rename" onClick={onRename} />
          <MenuItem icon={<Trash2 size={14} />} label="Delete"
                    danger onClick={onDelete} />
        </div>
      )}
    </li>
  );
}

function MenuItem({
  icon, label, onClick, danger = false,
}: {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex w-full items-center gap-2 px-3 py-1.5 text-left transition-colors ${
        danger
          ? "text-rose-700 hover:bg-rose-50"
          : "text-ink hover:bg-ivory-100"
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

function displayTitle(s: SessionListItem): string {
  if (s.title && s.title.trim()) return s.title;
  return "New chat";
}
