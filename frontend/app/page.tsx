"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { ArrowRight, MapPin, Cloud, Navigation, Zap } from "lucide-react";

const features = [
  { icon: Zap, label: "Self-critiquing AI", desc: "GPT-4 validates its own itineraries and auto-corrects errors" },
  { icon: Navigation, label: "Route Optimisation", desc: "TSP algorithm minimises total travel distance across stops" },
  { icon: MapPin, label: "Real Places", desc: "Google Places API — live ratings, addresses, and photos" },
  { icon: Cloud, label: "Live Weather", desc: "7-day forecasts woven into your itinerary planning" },
];

export default function Home() {
  const router = useRouter();
  const [destination, setDestination] = useState("");

  const handleStart = () => {
    if (destination.trim()) {
      router.push(`/chat?place=${encodeURIComponent(destination.trim())}`);
    } else {
      router.push("/chat");
    }
  };

  return (
    <main style={{ minHeight: "100vh", background: "var(--bg)", position: "relative", overflow: "hidden" }}>
      <div style={{
        position: "fixed", top: "-20%", left: "50%", transform: "translateX(-50%)",
        width: "800px", height: "500px",
        background: "radial-gradient(ellipse, rgba(79,172,254,0.07) 0%, transparent 70%)",
        pointerEvents: "none", zIndex: 0,
      }} />
      <nav style={{
        position: "fixed", top: 0, left: 0, right: 0, zIndex: 50,
        borderBottom: "1px solid var(--border)", backdropFilter: "blur(16px)",
        background: "rgba(8,8,8,0.7)", padding: "0 2rem", height: "56px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <span style={{ fontFamily: "Syne, sans-serif", fontWeight: 700, fontSize: "16px", letterSpacing: "-0.02em" }}>
          Travel Concierge
        </span>
        <button onClick={() => router.push("/chat")} style={{
          background: "var(--surface-3)", border: "1px solid var(--border)",
          color: "var(--text-primary)", padding: "6px 16px", borderRadius: "8px",
          fontSize: "13px", cursor: "pointer", fontFamily: "DM Sans, sans-serif",
        }}>Open App</button>
      </nav>
      <section style={{
        position: "relative", zIndex: 1, minHeight: "100vh",
        display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
        padding: "8rem 2rem 4rem", textAlign: "center",
      }}>
        <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}>
          <div style={{
            display: "inline-flex", alignItems: "center", gap: "6px",
            background: "var(--surface-2)", border: "1px solid var(--border)",
            borderRadius: "100px", padding: "4px 12px 4px 8px",
            fontSize: "12px", color: "var(--text-secondary)", marginBottom: "2rem",
          }}>
          </div>
          <h1 style={{
            fontSize: "clamp(42px, 7vw, 80px)", fontWeight: 800, letterSpacing: "-0.04em",
            lineHeight: 1.05, marginBottom: "1.5rem", fontFamily: "Syne, sans-serif",
          }}>
            Plan trips with an<br />
            <span style={{ background: "linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
              AI that thinks twice.
            </span>
          </h1>
          <p style={{ fontSize: "18px", color: "var(--text-secondary)", maxWidth: "520px", margin: "0 auto 3rem", lineHeight: 1.7, fontWeight: 300 }}>
            Personalised itineraries, optimised routes, and real-time weather —
            powered by a multi-agent system that critiques its own output.
          </p>
          <div style={{ display: "flex", gap: "8px", maxWidth: "480px", margin: "0 auto 1rem", flexWrap: "wrap", justifyContent: "center" }}>
            <input value={destination} onChange={e => setDestination(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleStart()}
              placeholder="Where do you want to go?"
              style={{
                flex: 1, minWidth: "240px", background: "var(--surface-2)",
                border: "1px solid var(--border)", borderRadius: "10px", padding: "12px 16px",
                color: "var(--text-primary)", fontSize: "15px", outline: "none", fontFamily: "DM Sans, sans-serif",
              }} />
            <button onClick={handleStart} style={{
              background: "linear-gradient(135deg, var(--accent), var(--accent-2))",
              border: "none", borderRadius: "10px", padding: "12px 20px", color: "#000",
              fontWeight: 600, fontSize: "14px", cursor: "pointer",
              display: "flex", alignItems: "center", gap: "6px",
              fontFamily: "DM Sans, sans-serif", whiteSpace: "nowrap",
            }}>
              Start Planning <ArrowRight size={15} />
            </button>
          </div>
          <p style={{ fontSize: "12px", color: "var(--text-muted)" }}>No signup required to try</p>
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 32 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.25, ease: [0.16, 1, 0.3, 1] }}
          style={{
            display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: "1px", maxWidth: "800px", width: "100%", margin: "5rem auto 0",
            background: "var(--border)", border: "1px solid var(--border)", borderRadius: "16px", overflow: "hidden",
          }}>
          {features.map(({ icon: Icon, label, desc }) => (
            <div key={label} style={{ background: "var(--surface)", padding: "1.75rem 1.5rem", transition: "background 0.2s" }}
              onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
              onMouseLeave={e => (e.currentTarget.style.background = "var(--surface)")}>
              <Icon size={18} color="var(--accent)" style={{ marginBottom: "0.75rem" }} />
              <div style={{ fontFamily: "Syne, sans-serif", fontWeight: 600, fontSize: "14px", marginBottom: "0.4rem" }}>{label}</div>
              <div style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.5 }}>{desc}</div>
            </div>
          ))}
        </motion.div>
      </section>
    </main>
  );
}
