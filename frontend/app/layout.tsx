import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Travel Concierge — AI Trip Planner",
  description: "Agentic AI travel planning. Personalised itineraries, route optimisation, real-time weather.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>{children}</body>
    </html>
  );
}
