// frontend/components/ui/Header.tsx
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { BrandLockup } from "@/components/ui/Brand";

const links = [
  { href: "/tutor",        label: "Tutor"        },
  { href: "/architecture", label: "Architecture" },
  { href: "/compare",      label: "Compare"      },
  { href: "/dashboard",    label: "Dashboard"    },
];

export function Header() {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-30 border-b border-ivory-200/70 bg-ivory-50/80 backdrop-blur supports-[backdrop-filter]:bg-ivory-50/60">
      <div className="flex w-full items-center justify-between px-4 py-3 sm:px-6 lg:px-8 xl:px-12">
        <Link
          href="/"
          aria-label="Socratic-OT home"
          className="ring-focus rounded transition-opacity hover:opacity-90"
        >
          <BrandLockup markSize={26} wordmarkSize={18} />
        </Link>

        <nav className="flex items-center gap-1 text-sm">
          {links.map((l) => {
            const active = path?.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`rounded-full px-3 py-1.5 transition-colors ring-focus ${
                  active
                    ? "bg-ink text-ivory-50"
                    : "text-ivory-600 hover:bg-ivory-100 hover:text-ink"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
