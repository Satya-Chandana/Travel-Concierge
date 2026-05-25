"use client";

import { useState, useEffect, useRef, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send, Plus, Trash2, MapPin, Cloud, Navigation,
  Compass, ChevronRight, Loader2, ArrowLeft
} from "lucide-react";
import Link from "next/link";
import { sendChat, getWeather, getPlaces, clearSession } from "@/lib/api";

// ── Types ─────────────────────────────────────────────────────────────────────

type Message = { role: "user" | "assistant"; content: string; intent?: string; id: string };
type Session = { id: string; name: string; messages: Message[]; place?: string };

const EXAMPLE_PROMPTS = (place: string) => [
  `Build me a 3-day itinerary for ${place}`,
  `Top attractions in ${place}?`,
  `Weather forecast for ${place} this week`,
  `Best dinner spots in ${place}`,
];

const INTENT_COLORS: Record<string, string> = {
  itinerary: "#4FACFE",
  places: "#43e97b",
  weather: "#fa709a",
  route: "#f6d365",
  restaurant: "#f093fb",
  general: "#888",
};

function genId() { return Math.random().toString(36).slice(2, 9); }

// ── Weather widget ────────────────────────────────────────────────────────────

function WeatherWidget({ place }: { place: string }) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!place) return;
    setLoading(true);
    getWeather(place, 3)
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [place]);

  if (!place) return null;

  return (
    <div style={{
      background: "var(--surface-2)", border: "1px solid var(--border)",
      borderRadius: "12px", padding: "1rem", marginBottom: "1rem",
    }}>
      <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "0.5rem" }}>
        Weather · {place}
      </div>
      {loading && <div style={{ color: "var(--text-muted)", fontSize: "13px" }}>Loading...</div>}
      {data?.days?.slice(0, 3).map((d: any) => (
        <div key={d.date} style={{
          display: "flex", justifyContent: "space-between", alignItems: "center",
          padding: "4px 0", borderBottom: "1px solid var(--border)", fontSize: "13px",
        }}>
          <span style={{ color: "var(--text-secondary)" }}>{d.date}</span>
          <span>{Math.round(d.max)}° / {Math.round(d.min)}°</span>
        </div>
      ))}
      {data?.formatted && !data?.days && (
        <div style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.5 }}>
          {data.formatted}
        </div>
      )}
    </div>
  );
}

// ── Message bubble ────────────────────────────────────────────────────────────

function MessageBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        marginBottom: "1rem",
        gap: "10px",
        alignItems: "flex-start",
      }}
    >
      {!isUser && (
        <div style={{
          width: "28px", height: "28px", borderRadius: "8px", flexShrink: 0, marginTop: "2px",
          background: "linear-gradient(135deg, var(--accent), var(--accent-2))",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <Compass size={14} color="#000" />
        </div>
      )}
      <div style={{
        maxWidth: "72%",
        background: isUser ? "var(--surface-3)" : "var(--surface-2)",
        border: `1px solid ${isUser ? "var(--border)" : "var(--border)"}`,
        borderRadius: isUser ? "16px 4px 16px 16px" : "4px 16px 16px 16px",
        padding: "0.75rem 1rem",
        fontSize: "14px", lineHeight: 1.7,
        color: "var(--text-primary)",
        whiteSpace: "pre-wrap",
      }}>
        {msg.content}
        {msg.intent && msg.intent !== "general" && (
          <div style={{
            marginTop: "6px", display: "inline-flex", alignItems: "center", gap: "4px",
            fontSize: "11px", color: INTENT_COLORS[msg.intent] || "#888",
            background: `${INTENT_COLORS[msg.intent]}18`,
            borderRadius: "4px", padding: "2px 6px",
          }}>
            {msg.intent}
          </div>
        )}
      </div>
    </motion.div>
  );
}

// ── Main chat component ───────────────────────────────────────────────────────

function ChatInner() {
  const searchParams = useSearchParams();
  const initialPlace = searchParams.get("place") || "";

  const [sessions, setSessions] = useState<Session[]>([
    { id: genId(), name: "Trip 1", messages: [], place: initialPlace }
  ]);
  const [activeId, setActiveId] = useState(sessions[0].id);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [place, setPlace] = useState(initialPlace);
  const [interests, setInterests] = useState("");
  const [onboarded, setOnboarded] = useState(!!initialPlace);
  const bottomRef = useRef<HTMLDivElement>(null);

  const active = sessions.find(s => s.id === activeId)!;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [active?.messages, loading]);

  const updateMessages = (msgs: Message[]) => {
    setSessions(prev => prev.map(s => s.id === activeId ? { ...s, messages: msgs } : s));
  };

  const handleSend = async (text?: string) => {
    const msg = (text || input).trim();
    if (!msg || loading) return;
    setInput("");

    const userMsg: Message = { role: "user", content: msg, id: genId() };
    const newMsgs = [...active.messages, userMsg];
    updateMessages(newMsgs);
    setLoading(true);

    try {
      const res = await sendChat(activeId, msg, place || active.place, interests);
      const assistantMsg: Message = {
        role: "assistant", content: res.response, intent: res.intent, id: genId()
      };
      updateMessages([...newMsgs, assistantMsg]);
    } catch {
      updateMessages([...newMsgs, {
        role: "assistant", content: "⚠️ Could not reach the server. Make sure the backend is running.",
        id: genId()
      }]);
    } finally {
      setLoading(false);
    }
  };

  const newSession = () => {
    const s: Session = { id: genId(), name: `Trip ${sessions.length + 1}`, messages: [] };
    setSessions(prev => [...prev, s]);
    setActiveId(s.id);
    setPlace("");
    setOnboarded(false);
  };

  const deleteSession = (id: string) => {
    clearSession(id).catch(() => {});
    const remaining = sessions.filter(s => s.id !== id);
    if (remaining.length === 0) {
      const fresh: Session = { id: genId(), name: "Trip 1", messages: [] };
      setSessions([fresh]);
      setActiveId(fresh.id);
    } else {
      setSessions(remaining);
      if (id === activeId) setActiveId(remaining[0].id);
    }
  };

  return (
    <div style={{ display: "flex", height: "100vh", background: "var(--bg)", overflow: "hidden" }}>

      {/* ── Sidebar ── */}
      <aside style={{
        width: "220px", flexShrink: 0,
        borderRight: "1px solid var(--border)",
        background: "var(--surface)",
        display: "flex", flexDirection: "column",
        padding: "1rem 0",
      }}>
        <div style={{ padding: "0 1rem 1rem", borderBottom: "1px solid var(--border)" }}>
          <Link href="/" style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--text-muted)", fontSize: "13px", textDecoration: "none", marginBottom: "1rem" }}>
            <ArrowLeft size={14} /> Home
          </Link>
          <div style={{ fontFamily: "Syne, sans-serif", fontWeight: 700, fontSize: "15px" }}>Travel Concierge</div>
        </div>

        <div style={{ padding: "0.75rem 1rem" }}>
          <button onClick={newSession} style={{
            width: "100%", display: "flex", alignItems: "center", gap: "8px",
            background: "var(--surface-3)", border: "1px solid var(--border)",
            borderRadius: "8px", padding: "8px 10px", color: "var(--text-secondary)",
            fontSize: "13px", cursor: "pointer", fontFamily: "DM Sans, sans-serif",
          }}>
            <Plus size={14} /> New Trip
          </button>
        </div>

        <div style={{ flex: 1, overflowY: "auto", padding: "0 0.5rem" }}>
          {sessions.map(s => (
            <div key={s.id}
              onClick={() => setActiveId(s.id)}
              style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
                padding: "8px 10px", borderRadius: "8px", cursor: "pointer",
                background: s.id === activeId ? "var(--surface-3)" : "transparent",
                color: s.id === activeId ? "var(--text-primary)" : "var(--text-secondary)",
                fontSize: "13px", marginBottom: "2px",
              }}>
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.name}</span>
              <button onClick={e => { e.stopPropagation(); deleteSession(s.id); }}
                style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", padding: "2px", opacity: 0, transition: "opacity 0.15s" }}
                onMouseEnter={e => (e.currentTarget.style.opacity = "1")}
                onMouseLeave={e => (e.currentTarget.style.opacity = "0")}>
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>

        {/* Weather widget */}
        <div style={{ padding: "0.75rem 1rem", borderTop: "1px solid var(--border)" }}>
          <WeatherWidget place={place || active?.place || ""} />
        </div>
      </aside>

      {/* ── Main ── */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

        {/* Header */}
        <div style={{
          padding: "0 1.5rem", height: "56px",
          borderBottom: "1px solid var(--border)",
          display: "flex", alignItems: "center", justifyContent: "space-between",
          flexShrink: 0,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            {place && (
              <div style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "13px", color: "var(--text-secondary)" }}>
                <MapPin size={13} color="var(--accent)" />
                {place}
              </div>
            )}
          </div>
          <div style={{ display: "flex", gap: "6px" }}>
            {[
              { icon: MapPin, label: "Places", prompt: `Top things to do in ${place || "this city"}` },
              { icon: Navigation, label: "Route", prompt: `Plan a walking route in ${place || "this city"}` },
              { icon: Cloud, label: "Weather", prompt: `7-day weather forecast for ${place || "this city"}` },
            ].map(({ icon: Icon, label, prompt }) => (
              <button key={label} onClick={() => handleSend(prompt)} style={{
                display: "flex", alignItems: "center", gap: "5px",
                background: "var(--surface-2)", border: "1px solid var(--border)",
                borderRadius: "7px", padding: "5px 10px",
                color: "var(--text-secondary)", fontSize: "12px", cursor: "pointer",
                fontFamily: "DM Sans, sans-serif",
              }}>
                <Icon size={12} /> {label}
              </button>
            ))}
          </div>
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflowY: "auto", padding: "1.5rem" }}>
          {/* Onboarding */}
          {!onboarded && (
            <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
              style={{ maxWidth: "520px", margin: "4rem auto", textAlign: "center" }}>
              <div style={{ fontFamily: "Syne, sans-serif", fontSize: "22px", fontWeight: 700, marginBottom: "0.5rem" }}>
                Where are you headed?
              </div>
              <p style={{ color: "var(--text-secondary)", fontSize: "14px", marginBottom: "1.5rem" }}>
                Tell me your destination and interests to get started.
              </p>
              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                <input value={place} onChange={e => setPlace(e.target.value)}
                  placeholder="Destination (e.g. Tokyo, Barcelona...)"
                  style={{
                    background: "var(--surface-2)", border: "1px solid var(--border)",
                    borderRadius: "10px", padding: "12px 14px", color: "var(--text-primary)",
                    fontSize: "14px", outline: "none", fontFamily: "DM Sans, sans-serif",
                  }} />
                <input value={interests} onChange={e => setInterests(e.target.value)}
                  placeholder="Interests (e.g. history, food, hiking...)"
                  style={{
                    background: "var(--surface-2)", border: "1px solid var(--border)",
                    borderRadius: "10px", padding: "12px 14px", color: "var(--text-primary)",
                    fontSize: "14px", outline: "none", fontFamily: "DM Sans, sans-serif",
                  }} />
                <button onClick={() => place && setOnboarded(true)} style={{
                  background: "linear-gradient(135deg, var(--accent), var(--accent-2))",
                  border: "none", borderRadius: "10px", padding: "12px",
                  color: "#000", fontWeight: 600, fontSize: "14px", cursor: "pointer",
                  display: "flex", alignItems: "center", justifyContent: "center", gap: "6px",
                  fontFamily: "DM Sans, sans-serif",
                }}>
                  Continue <ChevronRight size={15} />
                </button>
              </div>
            </motion.div>
          )}

          {/* Example prompts */}
          {onboarded && active.messages.length === 0 && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              style={{ maxWidth: "600px", margin: "3rem auto" }}>
              <div style={{ fontFamily: "Syne, sans-serif", fontSize: "18px", fontWeight: 700, marginBottom: "1.25rem", textAlign: "center" }}>
                Planning a trip to {place} ✈️
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
                {EXAMPLE_PROMPTS(place).map(p => (
                  <button key={p} onClick={() => handleSend(p)} style={{
                    background: "var(--surface-2)", border: "1px solid var(--border)",
                    borderRadius: "10px", padding: "12px 14px", textAlign: "left",
                    color: "var(--text-secondary)", fontSize: "13px", cursor: "pointer",
                    fontFamily: "DM Sans, sans-serif", lineHeight: 1.4,
                    transition: "border-color 0.15s, color 0.15s",
                  }}
                    onMouseEnter={e => { e.currentTarget.style.borderColor = "var(--border-hover)"; e.currentTarget.style.color = "var(--text-primary)"; }}
                    onMouseLeave={e => { e.currentTarget.style.borderColor = "var(--border)"; e.currentTarget.style.color = "var(--text-secondary)"; }}>
                    {p}
                  </button>
                ))}
              </div>
            </motion.div>
          )}

          {/* Messages */}
          <AnimatePresence initial={false}>
            {active.messages.map(msg => (
              <MessageBubble key={msg.id} msg={msg} />
            ))}
          </AnimatePresence>

          {/* Typing indicator */}
          {loading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "1rem" }}>
              <div style={{
                width: "28px", height: "28px", borderRadius: "8px",
                background: "linear-gradient(135deg, var(--accent), var(--accent-2))",
                display: "flex", alignItems: "center", justifyContent: "center",
              }}>
                <Compass size={14} color="#000" />
              </div>
              <div style={{
                background: "var(--surface-2)", border: "1px solid var(--border)",
                borderRadius: "4px 16px 16px 16px", padding: "12px 16px",
                display: "flex", gap: "4px", alignItems: "center",
              }}>
                {[0, 1, 2].map(i => (
                  <motion.div key={i}
                    animate={{ opacity: [0.3, 1, 0.3] }}
                    transition={{ duration: 1.2, delay: i * 0.2, repeat: Infinity }}
                    style={{ width: "6px", height: "6px", borderRadius: "50%", background: "var(--accent)" }} />
                ))}
              </div>
            </motion.div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div style={{
          padding: "1rem 1.5rem",
          borderTop: "1px solid var(--border)",
          background: "var(--surface)",
          flexShrink: 0,
        }}>
          <div style={{
            display: "flex", gap: "8px", alignItems: "flex-end",
            background: "var(--surface-2)", border: "1px solid var(--border)",
            borderRadius: "12px", padding: "8px 8px 8px 14px",
          }}>
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
              placeholder={onboarded ? `Ask anything about ${place || "your trip"}...` : "Set a destination first"}
              disabled={!onboarded || loading}
              rows={1}
              style={{
                flex: 1, background: "none", border: "none", outline: "none",
                color: "var(--text-primary)", fontSize: "14px", resize: "none",
                fontFamily: "DM Sans, sans-serif", lineHeight: 1.5, maxHeight: "120px",
              }}
            />
            <button onClick={() => handleSend()} disabled={!input.trim() || loading || !onboarded}
              style={{
                width: "34px", height: "34px", borderRadius: "8px", flexShrink: 0,
                background: input.trim() && !loading && onboarded
                  ? "linear-gradient(135deg, var(--accent), var(--accent-2))"
                  : "var(--surface-3)",
                border: "none", cursor: input.trim() && !loading && onboarded ? "pointer" : "default",
                display: "flex", alignItems: "center", justifyContent: "center",
                transition: "background 0.2s",
              }}>
              {loading
                ? <Loader2 size={15} color="var(--text-muted)" style={{ animation: "spin 1s linear infinite" }} />
                : <Send size={15} color={input.trim() && onboarded ? "#000" : "var(--text-muted)"} />
              }
            </button>
          </div>
          <p style={{ fontSize: "11px", color: "var(--text-muted)", textAlign: "center", marginTop: "6px" }}>
            Enter to send · Shift+Enter for new line
          </p>
        </div>
      </main>

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        textarea::placeholder { color: var(--text-muted); }
        input::placeholder { color: var(--text-muted); }
      `}</style>
    </div>
  );
}

export default function ChatPage() {
  return (
    <Suspense>
      <ChatInner />
    </Suspense>
  );
}
