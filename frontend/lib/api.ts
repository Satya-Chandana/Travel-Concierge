const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function sendChat(sessionId: string, message: string, place?: string, interests?: string) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message, place, interests }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ response: string; intent: string; session_id: string }>;
}

export async function getWeather(location: string, days = 3) {
  const res = await fetch(`${API_BASE}/api/weather`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ location, days }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getPlaces(location: string, interest: string) {
  const res = await fetch(`${API_BASE}/api/places`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ location, interest }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function clearSession(sessionId: string) {
  await fetch(`${API_BASE}/api/session/${sessionId}`, { method: "DELETE" });
}

export async function getItinerary(sessionId: string, location: string, interests: string, numDays: number) {
  const res = await fetch(`${API_BASE}/api/itinerary`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, location, interests, num_days: numDays }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function sendFeedback(sessionId: string, placeName: string, placeTypes: string[], features: number[], accepted: boolean) {
  await fetch(`${API_BASE}/api/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, place_name: placeName, place_types: placeTypes, features, accepted }),
  });
}
