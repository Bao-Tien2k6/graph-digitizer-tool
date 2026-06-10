/** Thin fetch wrappers around the FastAPI backend. */

async function jsonOrThrow(response) {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || JSON.stringify(body);
    } catch (_e) {
      // body wasn't JSON
    }
    throw new Error(`${response.status} ${detail}`);
  }
  return response.json();
}

export async function digitize(file) {
  const form = new FormData();
  form.append('image', file);
  const r = await fetch('/api/digitize', { method: 'POST', body: form });
  return jsonOrThrow(r);
}

export async function calibrate(sessionId, xTicks, yTicks) {
  const r = await fetch('/api/calibrate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      x_ticks: xTicks,
      y_ticks: yTicks,
    }),
  });
  return jsonOrThrow(r);
}

export async function deleteSession(sessionId) {
  if (!sessionId) return;
  await fetch(`/api/session/${sessionId}`, { method: 'DELETE' });
}

export function imageUrl(sessionId) {
  return `/api/image/${sessionId}`;
}
