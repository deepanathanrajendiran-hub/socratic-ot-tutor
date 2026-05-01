import type { Metadata } from "next";
import localFont from "next/font/local";
import { Source_Serif_4 } from "next/font/google";
import "./globals.css";
import { Header } from "@/components/ui/Header";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});
// Tiempos-adjacent open serif — used for the wordmark, page titles, and
// the assistant's prose blocks. Locked to 400/500/600/700 (regular,
// medium, semibold, bold) plus a 400 italic.
const sourceSerif = Source_Serif_4({
  subsets: ["latin"],
  variable: "--font-source-serif",
  weight: ["400", "500", "600", "700"],
  style: ["normal", "italic"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Socratic·OT — AI tutor for occupational therapy anatomy",
  description: "Socratic and Study modes for OT anatomy and neuroscience.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${sourceSerif.variable} min-h-screen bg-ivory-50 text-ink font-sans antialiased`}
      >
        <Header />
        <main className="w-full px-4 py-8 sm:px-6 md:py-10 lg:px-8 xl:px-12">
          {children}
        </main>
      </body>
    </html>
  );
}
